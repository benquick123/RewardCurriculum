import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
import copy
import stable_baselines3
from stable_baselines3.common.type_aliases import ReplayBufferSamples, DictReplayBufferSamples, TensorDict
from stable_baselines3.common.vec_env import VecNormalize


class ReplayBuffer(stable_baselines3.common.buffers.ReplayBuffer):
    
    def __init__(self, *args, reward_dim=1, **kwargs):
        kwargs["n_envs"] = 1
        super().__init__(*args, **kwargs)
        
        self.reward_dim = reward_dim
        self.rewards = np.zeros((self.buffer_size, self.n_envs, self.reward_dim), dtype=np.float32)
        self.reward_weights = np.zeros((self.buffer_size, self.n_envs, self.reward_dim), dtype=np.float32)
        self.action_ps = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        obs = obs.reshape((len(obs), self.n_envs, *self.obs_shape))
        next_obs = next_obs.reshape((len(next_obs), self.n_envs, *self.obs_shape))
        reward = reward.reshape(len(reward), self.n_envs, self.reward_dim)
        done = done.reshape(len(done), 1)

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((len(action), self.n_envs, self.action_dim))

        buffer_indices = np.arange(self.pos, self.pos + len(obs)) % self.buffer_size

        # Copy to avoid modification by reference
        self.observations[buffer_indices] = np.array(obs).copy()

        if self.optimize_memory_usage:
            self.observations[(buffer_indices + 1) % self.buffer_size] = np.array(next_obs).copy()
        else:
            self.next_observations[buffer_indices] = np.array(next_obs).copy()

        self.actions[buffer_indices] = np.array(action).copy()
        self.rewards[buffer_indices] = np.array(reward).copy()
        self.dones[buffer_indices] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[buffer_indices] = np.array([info.get("TimeLimit.truncated", False) for info in infos]).reshape(len(infos), 1)

        self.pos += len(obs)
        if self.pos >= self.buffer_size:
            self.full = True
            self.pos %= self.buffer_size

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, self.reward_dim), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
    

class HerReplayBuffer(stable_baselines3.her.HerReplayBuffer):
    
    def __init__(self, *args, scheduler, **kwargs):
        assert kwargs["n_envs"] == 1
        super().__init__(*args, **kwargs)
        
        self.scheduler = scheduler
        self.reward_dim = self.scheduler.reward_dim
        self.rewards = np.zeros((self.buffer_size, self.n_envs, self.reward_dim), dtype=np.float32)
        self.reward_weights = np.zeros((self.buffer_size, self.n_envs, self.reward_dim), dtype=np.float32)
        
    def add(
        self,
        obs: TensorDict,
        next_obs: TensorDict,
        action: np.ndarray,
        reward: np.ndarray,
        reward_weights: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        
        self.reward_weights[self.pos] = reward_weights
        super().add(obs, next_obs, action, reward, done, infos)
        
    def _get_real_samples(
        self,
        batch_indices: np.ndarray,
        env_indices: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:
        """
        Get the samples corresponding to the batch and environment indices.

        :param batch_indices: Indices of the transitions
        :param env_indices: Indices of the envrionments
        :param env: associated gym VecEnv to normalize the
            observations/rewards when sampling, defaults to None
        :return: Samples
        """
        # Normalize if needed and remove extra dimension (we are using only one env for now)
        obs_ = self._normalize_obs({key: np.array(obs[batch_indices, env_indices, :]) for key, obs in self.observations.items()}, env)
        next_obs_ = self._normalize_obs(
            {key: np.array(obs[batch_indices, env_indices, :]) for key, obs in self.next_observations.items()}, env
        )
        
        reward_weights = self.reward_weights[batch_indices, env_indices].reshape(-1, self.reward_dim)
        rewards = self.rewards[batch_indices, env_indices].reshape(-1, self.reward_dim)
        rewards = rewards.reshape(-1, 1, self.reward_dim) @ reward_weights.reshape(-1, 1, self.reward_dim).swapaxes(1, 2)
        rewards = rewards.reshape(-1, 1)
        
        obs_["observation"] = np.concatenate((obs_["observation"], reward_weights), axis=-1)
        next_obs_["observation"] = np.concatenate((next_obs_["observation"], reward_weights), axis=-1)

        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_indices, env_indices]),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_torch(
                self.dones[batch_indices, env_indices] * (1 - self.timeouts[batch_indices, env_indices])
            ).reshape(-1, 1),
            rewards=self.to_torch(self._normalize_reward(self.rewards[batch_indices, env_indices].reshape(-1, self.reward_dim), env)),
        )
        
    def _get_virtual_samples(
        self,
        batch_indices: np.ndarray,
        env_indices: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:
        """
        Get the samples, sample new desired goals and compute new rewards.

        :param batch_indices: Indices of the transitions
        :param env_indices: Indices of the envrionments
        :param env: associated gym VecEnv to normalize the
            observations/rewards when sampling, defaults to None
        :return: Samples, with new desired goals and new rewards
        """
        # Get infos and obs
        obs = {key: np.array(obs[batch_indices, env_indices, :]) for key, obs in self.observations.items()}
        next_obs = {key: np.array(obs[batch_indices, env_indices, :]) for key, obs in self.next_observations.items()}
        if self.copy_info_dict:
            # The copy may cause a slow down
            infos = copy.deepcopy(self.infos[batch_indices, env_indices])
        else:
            infos = [{} for _ in range(len(batch_indices))]
        # Sample and set new goals
        new_goals = self._sample_goals(batch_indices, env_indices)
        obs["desired_goal"] = new_goals
        # The desired goal for the next observation must be the same as the previous one
        next_obs["desired_goal"] = new_goals

        # Compute new reward
        rewards = self.env.env_method(
            "compute_reward",
            # the new state depends on the previous state and action
            # s_{t+1} = f(s_t, a_t)
            # so the next achieved_goal depends also on the previous state and action
            # because we are in a GoalEnv:
            # r_t = reward(s_t, a_t) = reward(next_achieved_goal, desired_goal)
            # therefore we have to use next_obs["achieved_goal"] and not obs["achieved_goal"]
            next_obs["achieved_goal"],
            # here we use the new desired goal
            obs["desired_goal"],
            infos,
            # we use the method of the first environment assuming that all environments are identical.
            indices=[0],
        )
        rewards = rewards[0].astype(np.float32)  # env_method returns a list containing one element
        
        reward_weights = self.scheduler.sample_batch(len(batch_indices))
        
        rewards = rewards.reshape(-1, 1, self.reward_dim) @ reward_weights.reshape(-1, 1, self.reward_dim).swapaxes(1, 2)
        rewards = rewards.reshape(-1, 1)
        
        obs["observation"] = np.concatenate((obs["observation"], reward_weights), axis=-1)
        next_obs["observation"] = np.concatenate((next_obs["observation"], reward_weights), axis=-1)
        
        obs = self._normalize_obs(obs, env)
        next_obs = self._normalize_obs(next_obs, env)

        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs.items()}

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_indices, env_indices]),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_torch(
                self.dones[batch_indices, env_indices] * (1 - self.timeouts[batch_indices, env_indices])
            ).reshape(-1, 1),
            rewards=self.to_torch(self._normalize_reward(rewards.reshape(-1, self.reward_dim), env)),
        )
        
    