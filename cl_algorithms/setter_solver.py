import numpy as np
import torch as th
from torch import nn, optim
from torch.nn import functional as F
import torchrl

from cl_algorithms.scheduler import Scheduler


class HighwayBlock(nn.Module):
    
    def __init__(self, hidden_size):
        super(HighwayBlock, self).__init__()

        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, 2 * hidden_size)
    
    def forward(self, x):
        linear_out = self.linear(x) # (batch_size, hidden_dim) -> (batch_size, 2 * hidden_dim)
        to_tanh, to_gate = linear_out.split(self.hidden_size, dim=-1)
        return x + (F.tanh(to_tanh) - x) * F.sigmoid(to_gate)
    

class Highway(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, num_blocks):
        super(Highway, self).__init__()
        
        modules = []
        modules.append(nn.Linear(input_size, hidden_size))
        modules.append(nn.Tanh())
        
        for _ in range(num_blocks):
            modules.append(HighwayBlock(hidden_size))
            
        modules.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.net(x)
        

class CouplingLayer(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, num_blocks):
        super(CouplingLayer, self).__init__()
        
        self.scale_net = nn.Sequential(
            Highway(input_size, hidden_size=hidden_size, output_size=output_size, num_blocks=num_blocks),
            nn.Softplus()
        )
        self.shift_net = Highway(input_size, hidden_size=hidden_size, output_size=output_size, num_blocks=num_blocks)
        
    def forward(self, z0, z1, log_p, condition, reverse=False):
        condition = condition.reshape(-1, 1) # [?] -> [batch_size, 1]
        _z0 = th.concat([z0, condition], dim=-1)
        
        scale = self.scale_net(_z0) + 1e-4
        shift = self.shift_net(_z0)
        
        if reverse:
            z1 = (z1 - shift) / scale
        else:
            z1 = z1 * scale + shift
        
        log_p -= scale.log().sum(dim=-1)
        
        return z1, log_p
    
    
class CouplingBlock(nn.Module):
    
    def __init__(self, total_input_size, hidden_size, num_blocks):
        super(CouplingBlock, self).__init__()
        
        self.module_list = nn.ModuleList([
            CouplingLayer(input_size=total_input_size // 2 + 1, 
                          hidden_size=hidden_size, 
                          output_size=total_input_size - total_input_size // 2, 
                          num_blocks=num_blocks),
            CouplingLayer(input_size=total_input_size - total_input_size // 2 + 1, 
                          hidden_size=hidden_size, 
                          output_size=total_input_size // 2,
                          num_blocks=num_blocks),
            CouplingLayer(input_size=total_input_size // 2 + 1, 
                          hidden_size=hidden_size, 
                          output_size=total_input_size - total_input_size // 2, 
                          num_blocks=num_blocks),
        ])
        self.modes = ["first", "last", "interlace"]
        
    def forward(self, z, log_p, condition, reverse=False):
        
        modules = self.module_list
        modes = self.modes
        if reverse:
            modules = modules[::-1]
            modes = modes[::-1]
        
        for module, mode in zip(modules, modes):
            z0, z1 = self._split(z, mode)
            
            if mode == "first":
                z_out, log_p = module(z0, z1, log_p, condition, reverse=reverse)
                z = th.concat([z0, z_out], dim=-1)
            elif mode == "last":
                z_out, log_p = module(z1, z0, log_p, condition, reverse=reverse)
                z = th.concat([z_out, z1], dim=-1)
            elif mode == "interlace":
                z_out, log_p = module(z1, z0, log_p, condition, reverse=reverse)
                z = self._interlace(z_out, z1)
            else:
                raise NotImplementedError
        
        return z, log_p
        
    def _split(self, x, mode):
        assert mode in {"first", "last", "interlace"}
        if mode in {"first", "last"}:
            x0 = x[:, :(x.shape[1] // 2)]
            x1 = x[:, (x.shape[1] // 2):]
        elif mode == "interlace":
            x0 = x[:, ::2]
            x1 = x[:, 1::2]
            
        return x0, x1
        
    def _interlace(self, x0, x1):
        smaller_index = np.argmin((x0.shape[1], x1.shape[1]))
        # there are only two arguments, so index is either 0 or 1
        larger_index = 1 - smaller_index
        smaller_size = (x0.shape[1], x1.shape[1])[smaller_index]
        
        x = th.stack([x0[:, :smaller_size], x1[:, :smaller_size]], dim=2).reshape(x0.shape[0], -1)
        x = th.concat([x, (x0, x1)[larger_index][:, smaller_size:]], dim=-1)
        return x
        

class Judge(nn.Module):
    
    def __init__(self, input_size, hidden_size=64, num_blocks=3, hidden_activation=nn.ReLU, output_activation=nn.Sigmoid):
        super().__init__()
        
        modules = []
        modules.append(nn.Linear(input_size, hidden_size))
        modules.append(hidden_activation())
        
        for _ in range(num_blocks - 1):
            modules.append(nn.Linear(hidden_size, hidden_size))
            modules.append(hidden_activation())
            
        modules.append(nn.Linear(hidden_size, 1))
        modules.append(output_activation())

        self.net = nn.Sequential(*modules)
    
    def forward(self, x):
        return self.net(x)
    
    
class Setter(nn.Module):
    
    def __init__(self, input_size, hidden_size=128, num_blocks=3, num_layers_per_block=3):
        super(Setter, self).__init__()
        
        modules = []
        for _ in range(num_blocks):
            modules.append(CouplingBlock(input_size, hidden_size=hidden_size, num_blocks=num_layers_per_block))
        self.module_list = nn.ModuleList(modules)
    
    def forward(self, x, condition, reverse=False):
        
        modules = self.module_list
        if reverse:
            # get sigmoid log_p
            log_p = -(self._log_sigmoid(x) + self._log_sigmoid(-x)).sum(dim=-1)
            # do the inverse sigmoid
            x = th.log(x / (1 - x))
            modules = modules[::-1]
        else:
            log_p = -0.5 * (x ** 2).sum(dim=-1)
        
        # traverse all the modules
        for module in modules:
            x, log_p = module(x, log_p, condition, reverse=reverse)
            
        if reverse:
            log_p -= 0.5 * (x ** 2).sum(dim=-1)
        else:
            # get sigmoid log_p
            log_p -= (self._log_sigmoid(x) + self._log_sigmoid(-x)).sum(dim=-1)
            # push through sigmoid fn
            x = F.sigmoid(x)
            
        return x, log_p
    
    def _log_sigmoid(self, x):
        return -F.softplus(-x)


class GoalDiscriminator(nn.Module):
    
    def __init__(self, input_size, hidden_size=64, hidden_activation=nn.ReLU, num_blocks=3):
        super(GoalDiscriminator, self).__init__()
        
        modules = []
        modules.append(nn.Linear(input_size, hidden_size))
        modules.append(hidden_activation())
        
        for _ in range(num_blocks - 1):
            modules.append(nn.Linear(hidden_size, hidden_size))
            modules.append(hidden_activation())
            
        modules.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.net(x)


class SetterSolver(Scheduler):
    
    ONE = th.tensor([1])
    MINUS_ONE = th.tensor([-1])
    
    def __init__(self, reward_dim, 
                 tau=10, 
                 seed=None, 
                 update_frequency=20, 
                 sampling_strategy="past",
                 success_threshold=5.0,
                 update_weights_frequency=1,
                 n_update_iterations=100,
                 goal_noise=0.05,
                 judge_hidden_size=64, setter_hidden_size=64, discriminator_hidden_size=64, 
                 goal_discriminator_weight=5):
        super(SetterSolver, self).__init__(reward_dim, tau=tau, seed=seed, update_frequency=update_frequency)
        
        assert sampling_strategy in {"past", "random"}
        self.sampling_strategy = sampling_strategy
        
        self.success_threshold = success_threshold
        
        self.judge = Judge(self.reward_dim, hidden_size=judge_hidden_size, output_activation=nn.Identity)
        self.judge_optimizer = optim.AdamW(self.judge.parameters(), lr=3e-4)
        self.judge_loss_fn = nn.BCEWithLogitsLoss()
        
        # setter_loss_noise_ub = 0.05
        self.setter = Setter(reward_dim, hidden_size=setter_hidden_size)
        self.setter_optimizer = optim.AdamW(self.setter.parameters(), lr=3e-4)
        self.feasibility_loss_fn = nn.MSELoss()
        
        target_mu = th.ones((reward_dim, )) * -1
        target_mu[-1] = 1
        # self.target_distribution = th.distributions.Normal(target_mu, 0.1)
        self.target_distribution = torchrl.modules.TruncatedNormal(target_mu, th.tensor(0.1), upscale=1.0, min=0.0, max=1.0, tanh_loc=True)
        
        self.goal_discriminator = GoalDiscriminator(reward_dim, hidden_size=discriminator_hidden_size)
        self.goal_discriminator_optimizer = optim.AdamW(self.goal_discriminator.parameters(), lr=5e-05)
        self.goal_discriminator_weight = goal_discriminator_weight
        
        self.reward_buffer = []
        self.weighted_reward_buffer = []
        self.weight_buffer = []
        self.main_reward_buffer = []
        self.main_weighted_reward_buffer = []
        self.main_weight_buffer = []
        self.update_weights_frequency = update_weights_frequency
        self.n_update_iterations = n_update_iterations
        self.batch_size = update_frequency // update_weights_frequency
        self.goal_noise = goal_noise
        
        self.n_maybe_update = 0
        
        self.has_judge_been_trained = False
        
        if self.sampling_strategy == "past":
            self.all_weights_history = []
        
    def maybe_update(self, **kwargs):
        assert "episode_rewards" in kwargs
        
        self.reward_buffer.append(kwargs["episode_rewards"])
        weighted_reward = (kwargs["episode_rewards"].reshape(1, -1) @ self.get_current_weights().reshape(-1, 1)).item()
        # if weighted_reward > self.success_threshold:
        #     print(f"Weighted reward larger than {self.success_threshold}: {weighted_reward}")
        self.weighted_reward_buffer.append(weighted_reward)
        self.weight_buffer.append(self.get_current_weights())
        
        if (self.n_maybe_update + 1) % self.update_weights_frequency == 0:
            self.main_weighted_reward_buffer.append(np.mean(self.weighted_reward_buffer))
            self.main_reward_buffer.append(np.stack(self.reward_buffer, axis=0).mean(axis=0))
            self.main_weight_buffer.append(np.stack(self.weight_buffer, axis=0).mean(axis=0))
            
            # if self.main_weighted_reward_buffer[-1] > self.success_threshold:
            #     print(f"Mean weighted reward larger than {self.success_threshold}: {np.mean(self.weighted_reward_buffer)}")
            
            self.init_period()
        
        if (self.n_maybe_update + 1) % self.update_frequency == 0:
            
            self.update()
            
            self.main_reward_buffer = []
            self.main_weighted_reward_buffer = []
            self.main_weight_buffer = []
            
            was_curriculum_updated = True
        else:
            was_curriculum_updated = False
            
        self.n_maybe_update += 1
            
        return was_curriculum_updated
    
    def update(self):
        # print("-------------------------------------------------------------")
        self.main_reward_buffer = th.tensor(np.array(self.main_reward_buffer), dtype=th.float)
        self.main_weighted_reward_buffer = th.tensor(np.array(self.main_weighted_reward_buffer), dtype=th.float)
        self.main_weight_buffer = th.tensor(np.stack(self.main_weight_buffer, axis=0), dtype=th.float)
        
        # if th.any(self.main_weighted_reward_buffer > self.success_threshold):
        #     print("Successful trajectories in main_weighted_replay_buffer.")
        # else:
        #     print("Max mean weighted reward:", th.max(self.main_weighted_reward_buffer))
        
        for iteration in range(self.n_update_iterations):
            self._update_goal_discriminator()
            
            self._update_judge()
            
            self._update_setter(iteration)
    
    def _generate_successful_weights(self, max_retries=10):
        successful_weights = self.main_weight_buffer[self.main_weighted_reward_buffer > self.success_threshold]
        if len(successful_weights) >= self.batch_size:
            return successful_weights[:self.batch_size]
        
        for _ in range(max_retries):
            new_weight_batch = th.tensor(self.rng.random(size=(self.batch_size, self.reward_dim)), dtype=th.float)
            assert self.main_reward_buffer.shape == new_weight_batch.shape, f"{self.main_reward_buffer.shape} is different from {new_weight_batch.shape}"
             
            weighted_rewards = (self.main_reward_buffer * new_weight_batch).sum(dim=-1)
            new_weight_batch = new_weight_batch[weighted_rewards > self.success_threshold]
            
            successful_weights = th.cat((successful_weights, new_weight_batch), dim=0)
            
            if len(successful_weights) >= self.batch_size:
                # print("Successful weights lenght:", len(successful_weights))
                return successful_weights[:self.batch_size]
    
        return successful_weights
    
    def _update_goal_discriminator(self):
        # clamp parameters to a cube
        for p in self.goal_discriminator.parameters():
            p.data.clamp_(-0.1, 0.1)
            
        real_samples = self.target_distribution.sample((self.batch_size, ))
        with th.no_grad():
            feasibilities = th.tensor(self.rng.random(size=(self.batch_size, 1)), dtype=th.float)
            zs = th.tensor(self.rng.normal(0, 1, size=(self.batch_size, self.reward_dim)), dtype=th.float)
            fake_samples = self.setter(zs, condition=feasibilities)[0]
            
        self.goal_discriminator.zero_grad()
        
        # get real
        real_output = self.goal_discriminator(real_samples).mean().view(1)
        real_output.backward(SetterSolver.ONE)
        
        # get fake
        fake_output = self.goal_discriminator(fake_samples).mean().view(1)
        fake_output.backward(SetterSolver.MINUS_ONE)
        
        self.goal_discriminator_optimizer.step()
    
    def _update_judge(self):
        successes = (self.main_weighted_reward_buffer > self.success_threshold).clone().detach().to(th.float).reshape(-1, 1)
        if th.sum(successes) == 0:
            return
        
        self.judge.zero_grad()
        
        y = self.judge(self.main_weight_buffer)
        loss = self.judge_loss_fn(y, successes)
        loss.backward()
        self.judge_optimizer.step()
        self.has_judge_been_trained = True
    
    def _update_setter(self, iteration=0):
        self.setter.zero_grad()
        
        # validity:
        successful_weights = self._generate_successful_weights()
        if len(successful_weights) > 0:
            successful_feasibilities = th.tensor(self.rng.random(size=(len(successful_weights), 1)), dtype=th.float)
            noise = th.tensor(self.rng.random(size=(len(successful_weights), self.reward_dim)) * self.goal_noise - (self.goal_noise / 2), dtype=th.float)
            successful_weights += noise
            successful_weights = successful_weights.clip(1e-5, 1.0 - 1e-5)
            validity_loss = (-self.setter(successful_weights, successful_feasibilities, reverse=True)[1]).mean()
        else:
            if iteration == 0:
                print("No valid weights found.")
            validity_loss = 0
        
        feasibilities = th.tensor(self.rng.random(size=(self.batch_size, 1)), dtype=th.float)
        zs = th.tensor(self.rng.normal(0, 1, size=(self.batch_size, self.reward_dim)), dtype=th.float)
        predicted_weights, predicted_log_p = self.setter(zs, feasibilities, reverse=False)
        
        # feasibility:
        if self.has_judge_been_trained:
            predicted_feasibility_logits = self.judge(predicted_weights)
            feasibility_loss = self.feasibility_loss_fn(F.sigmoid(predicted_feasibility_logits), feasibilities)
            self.has_judge_been_trained = False
        else:
            feasibility_loss = 0
        
        # coverage:
        coverage_loss = predicted_log_p.mean()
        
        # desired distribution
        if self.goal_discriminator_weight > 0:
            desired_loss = self.goal_discriminator(predicted_weights).mean()
        else:
            desired_loss = 0
        
        loss = validity_loss + feasibility_loss + coverage_loss + self.goal_discriminator_weight * desired_loss
        loss.backward()
        self.setter_optimizer.step()
        with th.no_grad():
            if iteration + 1 == self.n_update_iterations or iteration == 0:
                print(f"Setter losses @iteration {iteration}: {validity_loss}, {feasibility_loss}, {coverage_loss}, {self.goal_discriminator_weight * desired_loss}")
    
    def get_current_weights(self):
        return super().get_current_weights()
    
    @th.no_grad()
    def sample(self):
        feasibility = th.tensor(self.rng.random(size=(1, 1)), dtype=th.float)
        z = th.tensor(self.rng.normal(0, 1, size=(1, self.reward_dim)), dtype=th.float)
        self.current_weights = self.setter(z, condition=feasibility)[0].detach().numpy().reshape(-1)
        
        if self.sampling_strategy == "past":
            self.all_weights_history.append(self.current_weights)
    
    @th.no_grad()
    def sample_batch(self, batch_size):
        if self.sampling_strategy == "past":
            sample_idxs = self.rng.integers(low=0, high=len(self.all_weights_history), size=batch_size)
            weight_batch = np.array([self.all_weights_history[idx] for idx in sample_idxs])
            return weight_batch
        else:
            feasibilities = th.tensor(self.rng.random(size=(batch_size, 1)), dtype=th.float)
            zs = th.tensor(self.rng.normal(0, 1, size=(batch_size, self.reward_dim)), dtype=th.float)
            return self.setter(zs, condition=feasibilities)[0].detach().numpy()
