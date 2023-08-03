from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import stable_baselines3
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.type_aliases import (MaybeCallback,
                                                   RolloutReturn, TrainFreq)
from stable_baselines3.common.utils import (polyak_update,
                                            should_collect_more_steps)
from stable_baselines3.common.vec_env import VecEnv
from torch.nn import functional as F

from cl_algorithms.single_task import SingleTask
from rl_algorithms.replay_buffer import ReplayBuffer
from utils.callbacks import ProgressBarCallback


class SAC(stable_baselines3.SAC):
    
    def __init__(self, policy, env, reward_dim=1, scheduler_class=SingleTask, scheduler_kwargs={}, **kwargs):
        kwargs["replay_buffer_class"] = ReplayBuffer
        if "replay_buffer_kwargs" not in kwargs:
            kwargs["replay_buffer_kwargs"] = dict()
        kwargs["replay_buffer_kwargs"]["reward_dim"] = reward_dim
        self.scheduler = scheduler_class(reward_dim=reward_dim, **scheduler_kwargs)
        super(SAC, self).__init__(policy, env, **kwargs)
    
    def _setup_model(self) -> None:
        super()._setup_model()
        
        self.policy = self.policy.to("cpu")
        del self.policy
        
        observation_space = spaces.Box(-np.inf, np.inf, shape=(self.env.observation_space.shape[0] + self.scheduler.reward_dim, ), dtype=np.float32)
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)
        
        self._create_aliases()
      
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        action_mini_experience_buffer = [[] for _ in range(env.num_envs)]
        obs_mini_experience_buffer = [[] for _ in range(env.num_envs)]
        last_obs_mini_experience_buffer = [[] for _ in range(env.num_envs)]
        reward_mini_experience_buffer = [[] for _ in range(env.num_envs)]
        done_mini_experience_buffer = [[] for _ in range(env.num_envs)]
        info_mini_experience_buffer = [[] for _ in range(env.num_envs)]
        
        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            for idx in range(env.num_envs):
                action_mini_experience_buffer[idx].append(buffer_actions[idx])
                last_obs_mini_experience_buffer[idx].append(self._last_obs[idx])
                obs_mini_experience_buffer[idx].append(new_obs[idx])
                reward_mini_experience_buffer[idx].append(rewards[idx])
                done_mini_experience_buffer[idx].append(dones[idx])
                info_mini_experience_buffer[idx].append(infos[idx])
            self._last_obs = new_obs
            
            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)
            
            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # This has to do with the _on_step method of the actual algo.
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    self._store_transition(replay_buffer, 
                                           np.stack(action_mini_experience_buffer[idx], axis=0), 
                                           np.stack(last_obs_mini_experience_buffer[idx], axis=0),
                                           np.stack(obs_mini_experience_buffer[idx], axis=0),
                                           np.stack(reward_mini_experience_buffer[idx], axis=0), 
                                           np.stack(done_mini_experience_buffer[idx], axis=0), 
                                           info_mini_experience_buffer[idx])
                    
                    self.num_timesteps += len(action_mini_experience_buffer[idx])
                    num_collected_steps += len(action_mini_experience_buffer[idx]) / env.num_envs
                    
                    action_mini_experience_buffer[idx] = []
                    obs_mini_experience_buffer[idx] = []
                    last_obs_mini_experience_buffer[idx] = []
                    reward_mini_experience_buffer[idx] = []
                    done_mini_experience_buffer[idx] = []
                    info_mini_experience_buffer[idx] = []
                    
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()
                    
        assert self.replay_buffer.full or (not self.replay_buffer.full and self.replay_buffer.pos == self.num_timesteps), "Replay buffer vs. collected number of steps mismatch."
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)
    
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
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
        
        current_reward_weights = self.scheduler.get_current_weights()
        current_reward_weights = current_reward_weights.reshape((1, -1))
        current_reward_weights = np.repeat(current_reward_weights, int(batch_size / self.scheduler.reward_dim), axis=0)
        current_reward_weights = th.tensor(current_reward_weights, dtype=th.float32).to(self.device)
        
        main_task_reward_weights = th.zeros((int(batch_size / self.scheduler.reward_dim), self.scheduler.reward_dim), dtype=th.float32).to(self.device)
        main_task_reward_weights[:, -1] = 1.0
        
        remaining_batch_size = batch_size - len(current_reward_weights) - len(main_task_reward_weights)

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            
            # observations = th.cat((replay_data.observations.repeat(2, 1), current_reward_weights), dim=1)
            # next_observations = th.cat((replay_data.next_observations.repeat(2, 1), current_reward_weights), dim=1)
            # rewards = replay_data.rewards.repeat(2, 1).reshape(batch_size * 2, 1, -1) @ current_reward_weights.reshape(batch_size * 2, 1, -1).mT
            # rewards = rewards.reshape(batch_size * 2, 1)
            
            # reward_weights = current_reward_weights
            reward_weights = th.tensor(self.scheduler.sample_batch(remaining_batch_size), dtype=th.float32).to(self.device)
            reward_weights = th.concatenate((reward_weights, current_reward_weights, main_task_reward_weights), dim=0)
            
            observations = th.cat((replay_data.observations, reward_weights), dim=1)
            next_observations = th.cat((replay_data.next_observations, reward_weights), dim=1)
            rewards = replay_data.rewards.reshape(batch_size, 1, -1) @ reward_weights.reshape(batch_size, 1, -1).mT
            rewards = rewards.reshape(batch_size, 1)

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(observations)
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
                next_actions, next_log_prob = self.actor.action_log_prob(next_observations)
                # Compute the next Q values: min over all critics targets
                critic_indices = th.randperm(self.policy.critic.n_critics)[:2]
                next_q_values = th.cat(self.critic_target(next_observations, next_actions), dim=1)
                next_q_values = next_q_values[:, critic_indices]
                
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(observations, replay_data.actions.to(th.float32))

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(self.critic(observations, actions_pi), dim=1)
            # min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            mean_qf_pi = th.mean(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - mean_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

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
        last_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:

        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            print("Case when observation normalization is used is not implemented.")
            raise NotImplementedError

        # Avoid modification by reference
        last_obs = deepcopy(last_obs)
        next_obs = deepcopy(new_obs)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]

        replay_buffer.add(
            last_obs,
            next_obs,
            buffer_action,
            reward,
            dones,
            infos,
        )
        
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
            unscaled_action, _ = self.predict(self._last_obs, task=self.scheduler.get_current_weights(), deterministic=False)

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
        if task is None:
            assert observation.shape[-1] == (self.observation_space.shape[0] + self.scheduler.reward_dim), "Provide task or concatenate task with observation manually."
        else:
            assert observation.shape[-1] == self.observation_space.shape[0], "When task is given, the observation shape must match the observation space."
            task = task.reshape((-1, self.scheduler.reward_dim))
            if len(task) != len(observation):
                task = np.repeat(task, observation.shape[0], axis=0)
            observation = np.concatenate((observation, task), axis=-1)
            
        return self.policy.predict(observation, **kwargs)