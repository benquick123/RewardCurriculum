import random

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, MixtureSameFamily, Independent, MultivariateNormal
from torch.optim import AdamW, SGD
import numpy as np
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, DiagGaussianDistribution
from collections import OrderedDict
import pickle
from functools import partial
from copy import deepcopy
from utils.torch_distributions import GaussianTorchDistribution

from cl_algorithms.scheduler import Scheduler


def zero_grad(parameters):
    """
    Function used to set to zero the value of the gradient of a set
    of torch parameters.

    Args:
        parameters (list): list of parameters to be considered.

    """

    for p in parameters:
        if p.grad is not None:
           p.grad.detach_()
           p.grad.zero_()
           
           
def get_gradient(params):
    """
    Function used to get the value of the gradient of a set of
    torch parameters.

    Args:
        parameters (list): list of parameters to be considered.

    """
    views = []
    for p in params:
        if p.grad is None:
            view = p.new(p.numel()).zero_()
        else:
            view = p.grad.view(-1)
        views.append(view)
    return th.cat(views, 0)


def _fisher_vector_product_t(p, kl_fun, param_fun, cg_damping):
    kl = kl_fun()

    grads = th.autograd.grad(kl, param_fun(), create_graph=True, retain_graph=True)
    flat_grad_kl = th.cat([grad.view(-1) for grad in grads])

    kl_v = th.sum(flat_grad_kl * p)
    grads_v = th.autograd.grad(kl_v, param_fun(), create_graph=False, retain_graph=True)
    flat_grad_grad_kl = th.cat([grad.contiguous().view(-1) for grad in grads_v]).data

    return flat_grad_grad_kl + p * cg_damping


def _fisher_vector_product(p, kl_fun, param_fun, cg_damping, use_cuda=False):
    p_tensor = th.from_numpy(p)
    if use_cuda:
        p_tensor = p_tensor.cuda()

    return _fisher_vector_product_t(p_tensor, kl_fun, param_fun, cg_damping)


def _conjugate_gradient(b, kl_fun, param_fun, cg_damping, n_epochs_cg, cg_residual_tol, use_cuda=False):
    p = b.detach().cpu().numpy()
    r = b.detach().cpu().numpy()
    x = np.zeros_like(p)
    r2 = r.dot(r)

    for i in range(n_epochs_cg):
        z = _fisher_vector_product(p, kl_fun, param_fun, cg_damping, use_cuda=use_cuda).detach().cpu().numpy()
        v = r2 / p.dot(z)
        x += v * p
        r -= v * z
        r2_new = r.dot(r)
        mu = r2_new / r2
        p = r + mu * p

        r2 = r2_new
        if r2 < cg_residual_tol:
            break
    return x


def _line_search(prev_loss, stepdir, loss_fun, kl_fun, max_kl, param_fun, weight_setter, weight_getter,
                 cg_damping, n_epochs_line_search, use_cuda=False):
    # Compute optimal step size
    direction = _fisher_vector_product(stepdir, kl_fun, param_fun, cg_damping, use_cuda=use_cuda).detach().cpu().numpy()
    shs = .5 * stepdir.dot(direction)
    lm = np.sqrt(shs / max_kl)
    full_step = stepdir / lm
    stepsize = 1.

    # Save old policy parameters
    theta_old = weight_getter()

    # Perform Line search
    violation = True
    for _ in range(n_epochs_line_search):
        theta_new = theta_old + full_step * stepsize
        weight_setter(theta_new)

        new_loss = loss_fun()
        kl = kl_fun()
        improve = new_loss - prev_loss
        if kl <= max_kl * 1.5 or improve >= 0:
            violation = False
            break
        stepsize *= .95

    if violation:
        print("WARNING! KL-Divergence bound violation after linesearch")
        weight_setter(theta_old)


def cg_step(loss_fun, kl_fun, max_kl, param_fun, weight_setter, weight_getter, cg_damping, n_epochs_cg, cg_residual_tol, n_epochs_line_search, use_cuda=False):
    zero_grad(param_fun())
    loss = loss_fun()
    prev_loss = loss.item()
    loss.backward(retain_graph=True)

    g = get_gradient(param_fun())
    if np.linalg.norm(g) < 1e-10:
        print("Gradient norm smaller than 1e-10, skipping gradient step!")
        return
    else:
        if th.any(th.isnan(g)) or th.any(th.isinf(g)):
            raise RuntimeError("Nans and Infs in gradient")

        stepdir = _conjugate_gradient(g, kl_fun, param_fun, cg_damping, n_epochs_cg,
                                      cg_residual_tol, use_cuda=False)
        if np.any(np.isnan(stepdir)) or np.any(np.isinf(stepdir)):
            raise RuntimeError("Computation of conjugate gradient resulted in NaNs or Infs")

        _line_search(prev_loss, stepdir, loss_fun, kl_fun, max_kl, param_fun, weight_setter, weight_getter, cg_damping,
                     n_epochs_line_search, use_cuda=use_cuda)


class SPDL(Scheduler):
    
    def __init__(self, reward_dim, update_frequency, tau=20, seed=None, distributions=6, cl_batch_size=1024, nb_bootstrap=5, n_optim_iter=100, clip_ratio=0.2, alpha_proportion=1.0, initial_observation_buffer_size=1000, reward_buffer_size=2000, weight_buffer_size=5000):
        super(SPDL, self).__init__(reward_dim=reward_dim, tau=tau, seed=seed, update_frequency=update_frequency)
        
        assert nb_bootstrap < weight_buffer_size
        self.distributions = distributions
        self.nb_bootstrap = max(nb_bootstrap, distributions)
        self.clip_ratio = clip_ratio
        self.n_optim_iter = n_optim_iter
        self.alpha_proportion = alpha_proportion
        self.cl_batch_size = cl_batch_size
        self.reward_buffer_size = reward_buffer_size
        
        # self.mix_log_probs = (th.ones(self.distributions) * (1 / self.distributions)).requires_grad_(False)
        # self.mu = nn.Parameter(th.ones((1, reward_dim)))
        # self.log_std = nn.Parameter(th.zeros((1, reward_dim)))
        # self.optimizer = AdamW([self.mu, self.log_std], lr=0.0001)
        
        target_mu = th.zeros((reward_dim, ))
        target_mu[-1] = 10
        self.target_distribution = GaussianTorchDistribution(target_mu, GaussianTorchDistribution.flatten_matrix(th.eye(reward_dim)) * 0.1, use_cuda=False)
        
        # self.distribution = DiagGaussianDistribution(reward_dim)
        self.distribution = GaussianTorchDistribution(th.zeros(reward_dim), GaussianTorchDistribution.flatten_matrix(th.eye(reward_dim)), use_cuda=False)
        
        self.learner = None
        self.alpha = 0
        self.n_maybe_update = 0
        self.n_updates = 0
        self.weighted_rewards_history = []
        
        self.initial_observation_buffer_size = initial_observation_buffer_size
        self.initial_observation_buffer = list()
        
        self.weight_buffer_size = weight_buffer_size
        self.weight_buffer = []
        
    def maybe_update(self, **kwargs):
        assert "initial_observation" in kwargs
        assert "learner" in kwargs
        assert "episode_rewards" in kwargs
        
        self.initial_observation_buffer.append(kwargs["initial_observation"])
        if len(self.initial_observation_buffer) > self.initial_observation_buffer_size:
            self.initial_observation_buffer = self.initial_observation_buffer[1:]
            
        weighted_reward = (kwargs["episode_rewards"].reshape(1, -1) @ self.get_current_weights().reshape(-1, 1)).item()
        self.weighted_rewards_history.append(weighted_reward)
        if len(self.weighted_rewards_history) > self.reward_buffer_size:
            self.weighted_rewards_history = self.weighted_rewards_history[1:]
            
        self.weight_buffer.append(self.current_weights)
        if len(self.weight_buffer) > self.weight_buffer_size:
            self.weight_buffer = self.weight_buffer[1:]
            
        if (self.n_maybe_update + 1) % self.update_frequency == 0 and len(self.weight_buffer) > self.nb_bootstrap:            
            # do the update
            self.learner = kwargs["learner"]
            self.update()
            self.learner = None
            self.n_updates += 1
            was_curriculum_updated = True
        else:
            was_curriculum_updated = False
            
        self.init_period()
        self.n_maybe_update += 1
        
        return was_curriculum_updated
        
    def _approx_kl_div(self, log_prob, target_log_prob):
        # http://joschu.net/blog/kl-approx.html
        log_r = log_prob - target_log_prob
        return ((log_r.exp() - 1) - log_r).mean()
    
    def _compute_context_loss(self, cons_t, old_c_log_prob_t, c_val_t, alpha):
        con_ratio_t = th.exp(self.distribution.log_pdf_t(cons_t) - old_c_log_prob_t)
        kl_div = th.distributions.kl.kl_divergence(self.distribution.distribution_t, self.target_distribution.distribution_t)
        return th.mean(con_ratio_t * c_val_t - alpha * kl_div)
    
    def _compute_context_kl(self, old_context_dist):
        return th.distributions.kl.kl_divergence(old_context_dist.distribution_t, self.distribution.distribution_t)
    
    def _target_context_kl(self, numpy=True):
        kl_div = th.distributions.kl.kl_divergence(self.distribution.distribution_t, self.target_distribution.distribution_t).detach()
        if numpy:
            kl_div = kl_div.numpy()

        return kl_div
    
    def update(self):
        assert self.learner is not None
        
        # sample reward weights
        reward_weights = th.tensor(self.rng.choice(self.weight_buffer, size=self.cl_batch_size))
        
        # compute alpha
        if len(self.weighted_rewards_history) > self.nb_bootstrap * 10:
            with th.no_grad():
                rewards_mean = th.tensor(self.weighted_rewards_history).mean()
                target_dist_kl = self._target_context_kl()
                self.alpha = self.alpha_proportion * rewards_mean / (target_dist_kl + 1e-8)
                print("new alpha", self.alpha)
        
        # get initial state values
        initial_states = random.choices(self.initial_observation_buffer, k=self.cl_batch_size)
        if isinstance(initial_states[0], dict):
            concatenated_initial_states = OrderedDict()
            for element in initial_states:
                for key, value in element.items():
                    if key not in concatenated_initial_states:
                        concatenated_initial_states[key] = []
                    concatenated_initial_states[key].append(value)
            for key, value in concatenated_initial_states.items():
                concatenated_initial_states[key] = th.tensor(np.concatenate(value, axis=0)).to(self.learner.device)
                if key == "observation":
                    concatenated_initial_states[key] = th.concat((concatenated_initial_states[key], F.softmax(reward_weights, dim=-1).to(self.learner.device)), dim=1)
            initial_states = concatenated_initial_states
        else:
            initial_states = th.tensor(np.concatenate(initial_states, axis=0)).to(self.learner.device)
            initial_states = th.concat((initial_states, F.softmax(reward_weights).to(self.learner.device)), dim=1)
        
        # get values
        with th.no_grad():
            mean_actions = pickle.loads(pickle.dumps(self.learner.actor))(initial_states, deterministic=True).detach()
            values = pickle.loads(pickle.dumps(self.learner.critic_target))(initial_states, mean_actions).detach()
            values = th.mean(values, dim=1)
            values = values.to("cpu")
            values = values.mean(dim=1)
        
        prev_distribution = deepcopy(self.distribution)
        prev_log_probs = prev_distribution.log_pdf_t(reward_weights).detach()
        
        cg_parameters = {"n_epochs_line_search": 200, "n_epochs_cg": 10, "cg_damping": 1e-2, "cg_residual_tol": 1e-10}
        
        cg_step(partial(self._compute_context_loss, reward_weights, prev_log_probs, values, self.alpha), 
                partial(self._compute_context_kl, prev_distribution), 0.1,
                self.distribution.parameters, self.distribution.set_weights, 
                self.distribution.get_weights, **cg_parameters, use_cuda=False)
        
        # print("-----")
        # print(self.distribution.mean())
        # print(self.distribution.covariance_matrix())
        
        # with th.no_grad():
        #     import os
        #     os.makedirs("tmp_plots", exist_ok=True)
        #     from matplotlib import pyplot as plt
        #     # fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        #     # axes = np.reshape(axes, -1)
        #     # for reward_dim, ax in enumerate(axes):
        #     #     value_mean = values # .mean(dim=1)
        #     #     reward_w = reward_weights[:, reward_dim]
        #     #     ax.plot(value_mean, reward_w, ".", color="C0")
        #     #     ax.plot([value_mean.mean()], [reward_w.mean()], "o", color="C1")
        #     #     ax.add_patch(Ellipse(xy=(value_mean.mean(), reward_w.mean()), width=value_mean.std(), height=reward_w.std(), edgecolor="r", fc="r", alpha=0.1))
        #     
        #     fig = plt.figure(figsize=(10, 10))
        #     ax = fig.add_subplot(111, projection='3d')
        #     colors = ((values - values.min()) / (values.max() - values.min())).numpy()
        #     ax.scatter(reward_weights[:, 0], reward_weights[:, 1], reward_weights[:, 2], c=colors)
        #     # values = values_dist.sample((1024, ))
        #     # ax.scatter(values[:, 0], values[:, 1], values[:, 2])
        #     
        #     u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        #     soft_mu = self.distribution.mean()
        #     # print(soft_mu)
        #     x = soft_mu[0] + np.cos(u)*np.sin(v) * np.sqrt(self.distribution.covariance_matrix()[0, 0])
        #     y = soft_mu[1] + np.sin(u)*np.sin(v) * np.sqrt(self.distribution.covariance_matrix()[1, 1])
        #     z = soft_mu[2] + np.cos(v) * np.sqrt(self.distribution.covariance_matrix()[2, 2])
        #     ax.plot_wireframe(x, y, z, color="r", alpha=0.5)
        #     
        #     ax.set_xlabel("$w_0$")
        #     ax.set_ylabel("$w_1$")
        #     ax.set_zlabel("$w_2$")
        #     ax.view_init(30, 60)
        #         
        #     plt.savefig("tmp_plots/%010d.png" % self.n_updates)
        #     plt.close()
        # print("-----")
      
    @th.no_grad()  
    def get_current_weights(self):
        return F.softmax(self.current_weights, dim=-1).numpy()
    
    @th.no_grad()
    def sample(self):
        # if len(self.weighted_rewards_history) < self.nb_bootstrap or self.gmm is None:
        #     self.current_weights = self.rng.random(self.reward_dim)
        #     self.current_weights = self.current_weights / (self.current_weights.sum() + 1e-8)
        # else:
        self.current_weights = self.distribution.sample().detach()
        # self.current_weights = F.softmax(self.current_weights, dim=-1).numpy()
        # self.current_weights = self.distribution.actions_from_params(self.mu, self.log_std)
        # self.current_weights = self.current_weights[0].numpy()
            
        # normalize the weights
        # self.current_weights = np.exp(self.current_weights * self.tau) / np.exp(self.current_weights * self.tau).sum()
        # self.current_weights = np.clip(self.current_weights, 0.0, 1.0)
        # self.current_weights /= (self.current_weights.sum() + 1e-8)
        # self.current_weights -= self.current_weights.min()
        # print(self.current_weights)
        
    @th.no_grad()
    def sample_batch(self, batch_size):
        weight_batch = th.stack([self.distribution.sample() for _ in range(batch_size)], dim=0)
        weight_batch = F.softmax(weight_batch, dim=-1)
        # weight_batch = np.array(self.get_current_weights()).reshape(1, -1)
        # weight_batch = np.repeat(weight_batch, batch_size, axis=0)
        return weight_batch.numpy()
        
