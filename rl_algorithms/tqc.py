from copy import deepcopy
from functools import partial
from typing import ClassVar, Dict, Optional, Tuple, Type, Union, List, Any

import numpy as np
import sb3_contrib
import stable_baselines3
import torch as th
from gymnasium import spaces
from sb3_contrib.common.utils import quantile_huber_loss
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.utils import polyak_update

from cl_algorithms.single_task import SingleTask
from rl_algorithms.replay_buffer import HerReplayBuffer, ReplayBuffer
from rl_algorithms.tqc_policies import (CnnPolicy, MlpPolicy, MultiInputPolicy,
                                        TQCPolicy)
from utils.callbacks import ProgressBarCallback
from utils.retrospective_loss import retrospective_loss_fn


class TQC(sb3_contrib.TQC):
    
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: TQCPolicy
    
    def __init__(self, policy, env, use_uvfa=True, reward_dim=1, scheduler_class=SingleTask, scheduler_kwargs={}, use_retrospective_loss=False, **kwargs):
        self.use_uvfa = use_uvfa
        self.scheduler = scheduler_class(reward_dim=reward_dim, **scheduler_kwargs)
        kwargs["replay_buffer_class"] = HerReplayBuffer
        if "replay_buffer_kwargs" not in kwargs:
            kwargs["replay_buffer_kwargs"] = dict()
        # kwargs["replay_buffer_kwargs"]["reward_dim"] = reward_dim
        kwargs["replay_buffer_kwargs"]["scheduler"] = self.scheduler
        kwargs["replay_buffer_kwargs"]["use_uvfa"] = use_uvfa
        
        self.use_retrospective_loss = use_retrospective_loss
        super().__init__(policy, env, **kwargs)
        
    def _setup_model(self) -> None:
        super()._setup_model()
        
        self.policy = self.policy.to("cpu")
        del self.policy
        
        if isinstance(self.env.observation_space, spaces.Box):
            observation_size = self.env.observation_space.shape[0] + self.scheduler.reward_dim * self.use_uvfa
            
            observation_space = spaces.Box(-np.inf, np.inf, shape=(observation_size, ), dtype=np.float32)
        elif isinstance(self.env.observation_space, spaces.Dict):
            observation_space = deepcopy(self.env.observation_space)
            observation_size = self.env.observation_space["observation"].shape[0] + self.scheduler.reward_dim * self.use_uvfa
            observation_space["observation"] = spaces.Box(-np.inf, np.inf, shape=(observation_size, ), dtype=np.float32)
        
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            observation_space,
            self.action_space,
            self.lr_schedule,
            use_retrospective_loss=self.use_retrospective_loss,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)
        
        self._create_aliases()
        
        if self.use_retrospective_loss:
            self.critic_retro = self.policy.critic_retro
            
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        return super().train(gradient_steps, batch_size)
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            
            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute and cut quantiles at the next state
                # batch x nets x quantiles
                next_quantiles = self.critic_target(replay_data.next_observations, next_actions)

                # Sort and drop top k quantiles to control overestimation.
                n_target_quantiles = self.critic.quantiles_total - self.top_quantiles_to_drop_per_net * self.critic.n_critics
                next_quantiles, _ = th.sort(next_quantiles.reshape(batch_size, -1))
                next_quantiles = next_quantiles[:, :n_target_quantiles]

                # td error + entropy term
                target_quantiles = next_quantiles - ent_coef * next_log_prob.reshape(-1, 1)
                target_quantiles = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_quantiles
                # Make target_quantiles broadcastable to (batch_size, n_critics, n_target_quantiles).
                target_quantiles.unsqueeze_(dim=1)

            # Get current Quantile estimates using action from the replay buffer
            current_quantiles = self.critic(replay_data.observations, replay_data.actions.to(th.float32))
            # Compute critic loss, not summing over the quantile dimension as in the paper.
            critic_loss = quantile_huber_loss(current_quantiles, target_quantiles, sum_over_quantiles=False)
            critic_losses.append(critic_loss.item())
            
            if self._n_updates > 0 and gradient_step > 0 and self.use_retrospective_loss:
                if self._n_updates == 1:
                    print("Retrospective loss has not been tested with TQC yet! Probably doesn't work.")
                    
                with th.no_grad():
                    retro_quantiles = self.critic_retro(replay_data.observations, replay_data.actions.to(th.float32))
                    
                retrospective_loss = sum([
                    retrospective_loss_fn(
                        current_quantiles[:, i, :].unsqueeze(1),
                        retro_quantiles[:, i, :].unsqueeze(1),
                        target_quantiles,
                        K=2, scaled=True, distance=lambda: partial(quantile_huber_loss, sum_over_quantiles=False)
                    ) for i in range(self.critic.n_critics)
                ])

                
                critic_loss += retrospective_loss

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            qf_pi = self.critic(replay_data.observations, actions_pi).mean(dim=2).mean(dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see https://github.com/DLR-RM/stable-baselines3/issues/996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
                
        if self.use_retrospective_loss:
            self.critic_retro.load_state_dict(self.critic.state_dict())

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
    
    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        replay_buffer.add(
            self._last_original_obs,
            next_obs,
            buffer_action,
            reward_,
            self.scheduler.get_current_weights(),
            dones,
            infos,
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_
    
    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter        
            unscaled_action, _ = self.predict(deepcopy(self._last_obs), task=self.scheduler.get_current_weights(), deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action
            
    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False
    ) -> Tuple[int, BaseCallback]:
        
        total_timesteps, callback = super()._setup_learn(total_timesteps=total_timesteps, 
                             callback=callback, 
                             reset_num_timesteps=reset_num_timesteps, 
                             tb_log_name=tb_log_name, 
                             progress_bar=progress_bar)
        
        if progress_bar:
            for idx, _callback in enumerate(callback.callbacks):
                if isinstance(_callback, stable_baselines3.common.callbacks.ProgressBarCallback):
                    callback.callbacks[idx] = ProgressBarCallback()
        callback.init_callback(self)
            
        return total_timesteps, callback
    
    def predict(self, observation: np.ndarray, task: np.ndarray = None, **kwargs) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        observation_dim = observation.shape[-1] if isinstance(self.observation_space, spaces.Box) else self.observation_space["observation"].shape[-1]
        expected_observation_dim = self.observation_space.shape[0] if isinstance(self.observation_space, spaces.Box) else self.observation_space["observation"].shape[0]
        
        if self.use_uvfa:
            if task is None:
                assert observation_dim == (expected_observation_dim + self.scheduler.reward_dim), "Provide task or concatenate task with observation manually."
            else:
                assert observation_dim == expected_observation_dim, "When task is given, the observation shape must match the observation space."
                task = task.reshape((-1, self.scheduler.reward_dim))
                if len(task) != len(observation):
                    batch_size = observation.shape[0] if isinstance(observation, spaces.Box) else observation["observation"].shape[0]
                    task = np.repeat(task, batch_size, axis=0)
                    
                if isinstance(observation, spaces.Box):
                    observation = np.concatenate((observation, task), axis=-1)
                else:
                    observation["observation"] = np.concatenate((observation["observation"], task), axis=-1)
            
        return self.policy.predict(observation, **kwargs)