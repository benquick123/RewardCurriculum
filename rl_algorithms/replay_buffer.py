import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
import stable_baselines3
from stable_baselines3.common.type_aliases import ReplayBufferSamples
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