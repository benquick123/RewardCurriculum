import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from collections import defaultdict, OrderedDict
import copy
import stable_baselines3
from stable_baselines3.common.type_aliases import ReplayBufferSamples, DictReplayBufferSamples, TensorDict
from stable_baselines3.common.vec_env import VecNormalize


class ReplayBuffer(stable_baselines3.common.buffers.ReplayBuffer):
    
    def __init__(self, *args, scheduler=None, **kwargs):
        kwargs["n_envs"] = 1
        super().__init__(*args, **kwargs)
        
        self.scheduler = scheduler
        self.reward_dim = self.scheduler.reward_dim
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
    
    def __init__(self, *args, scheduler=None, use_uvfa=True, **kwargs):
        assert kwargs["n_envs"] == 1
        super().__init__(*args, **kwargs)
        
        self.scheduler = scheduler
        self.use_uvfa = use_uvfa
        self.reward_dim = self.scheduler.reward_dim
        self.rewards = np.zeros((self.buffer_size, self.n_envs, self.reward_dim), dtype=np.float32)
        self.cumulative_rewards = np.zeros((self.buffer_size, self.n_envs, self.reward_dim), dtype=np.float32)
        
        self.has_diversity_changed = True
        self.normalized_diversity = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.unique_return_counts = [defaultdict() for _ in range(self.reward_dim)]
        self.unique_returns = [set() for _ in range(self.reward_dim)]
        
    def add(
        self,
        obs: TensorDict,
        next_obs: TensorDict,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        
        super().add(obs, next_obs, action, reward, done, infos)
        
        # When episode ends, compute the discounted rewards for each of the objecties
        for env_idx in range(self.n_envs):
            if done[env_idx]:
                self._compute_episode_returns(env_idx)
                # Update the current episode start
                self._current_ep_start[env_idx] = self.pos
                
        self.has_diversity_changed = True
                            
    def _compute_episode_returns(self, env_idx):
        episode_start = self._current_ep_start[env_idx]
        episode_end = self.pos
        if episode_end < episode_start:
            episode_end += self.buffer_size
        episode_indices = np.arange(episode_start, episode_end) % self.buffer_size
        
        # hardcoded gamma for now
        powers = 0.95 ** np.arange(len(episode_indices))
        powers = np.repeat(np.expand_dims(powers, axis=1), self.reward_dim, axis=1)
        cumulative_reward = (powers * self.rewards[episode_indices, env_idx]).sum(axis=0)
        self.cumulative_rewards[episode_indices, env_idx] = cumulative_reward
        for reward_idx, (unique_counts, unique_returns) in enumerate(zip(self.unique_return_counts, self.unique_returns)):
            # add to the unique returns set
            return_value = cumulative_reward[reward_idx]
            unique_returns.add(return_value)
            
            # take care fo the reward counting.
            return_value = np.round(return_value, 1)
            # print(return_value)
            if return_value not in unique_counts:
                unique_counts[return_value] = 0
            unique_counts[return_value] += len(episode_indices)
        
    def _compute_episode_length(self, env_idx: int) -> None:
        episode_start = self._current_ep_start[env_idx]
        episode_end = self.pos
        if episode_end < episode_start:
            # Occurs when the buffer becomes full, the storage resumes at the
            # beginning of the buffer. This can happen in the middle of an episode.
            episode_end += self.buffer_size
        episode_indices = np.arange(episode_start, episode_end) % self.buffer_size
        # print(len(episode_indices))
        self.ep_length[episode_indices, env_idx] = episode_end - episode_start
        
    def _compute_single_reward_diversity(self, reward_idx, is_valid):
        # create a 2d array containing only populated indices, and matching the cummulative rewards for current reward_idx
        cum_reward_subset = self.cumulative_rewards[is_valid][:, reward_idx]
        
        # get array of indices that sort the cum_reward_subset.
        # make sure you use the "stable" argument.
        sorted_reward_indices = np.argsort(cum_reward_subset, kind="stable")
        inverse_sorted_reward_indices = np.argsort(sorted_reward_indices, kind="stable")
        
        # sort the cumulative rewards
        sorted_cum_reward_subset = cum_reward_subset[sorted_reward_indices]
        div_min = sorted_cum_reward_subset[0]
        div_max = sorted_cum_reward_subset[-1]
        if div_min == div_max:
            return np.zeros_like(cum_reward_subset)
        
        # get unique returns
        unique_counts = self.unique_return_counts[reward_idx]
        unique_returns = np.array(sorted(self.unique_returns[reward_idx]))
        if len(unique_returns) == 2:
            return np.zeros_like(cum_reward_subset)
        
        # ... and indices for these unique returns
        unique_return_indices = np.cumsum(sorted_cum_reward_subset[:-1] != sorted_cum_reward_subset[1:])
        unique_return_indices = np.concatenate([[0], unique_return_indices])
        prev_return_indices = (unique_return_indices-1).clip(min=0)
        next_return_indices = (unique_return_indices+1).clip(max=len(unique_returns)-1)
        # get diversisiest of sorted elements
        # take into account the frequiencies of the values? (e.g. zero is very common, but won't be sampled much if there is a cum_rew nearby.)
        diversities = unique_returns[next_return_indices] - unique_returns[prev_return_indices]
        diversities /= (div_max - div_min)
        
        # inverse sort the diversisties
        diversities = diversities[inverse_sorted_reward_indices]
        # diversities /= np.array([unique_counts[return_value] for return_value in cum_reward_subset.astype(float).round(1)])
        return diversities
        
    def _compute_diversity(self):
        is_valid = self.ep_length > 0
        self.normalized_diversity = np.zeros_like(self.normalized_diversity)
        for reward_idx in range(self.reward_dim):
            self.normalized_diversity[is_valid] += self._compute_single_reward_diversity(reward_idx, is_valid)
            
        self.normalized_diversity[is_valid] += 1
        self.normalized_diversity /= np.sum(self.normalized_diversity)
        
    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        """
        Sample elements from the replay buffer.

        :param batch_size: Number of element to sample
        :param env: Associated VecEnv to normalize the observations/rewards when sampling
        :return: Samples
        """
        # When the buffer is full, we rewrite on old episodes. We don't want to
        # sample incomplete episode transitions, so we have to eliminate some indexes.        
        is_valid = self.ep_length > 0
        if not np.any(is_valid):
            raise RuntimeError(
                "Unable to sample before the end of the first episode. We recommend choosing a value "
                "for learning_starts that is greater than the maximum number of timesteps in the environment."
            )
        
        # compute new diversity if neccessary (only recompute when new experience has been added)
        if False and self.has_diversity_changed:
            self.has_diversity_changed = False
            self._compute_diversity()
        
        # Get the indices of valid transitions
        # Example:
        # if is_valid = [[True, False, False], [True, False, True]],
        # is_valid has shape (buffer_size=2, n_envs=3)
        # then valid_indices = [0, 3, 5]
        # they correspond to is_valid[0, 0], is_valid[1, 0] and is_valid[1, 2]
        # or in numpy format ([rows], [columns]): (array([0, 1, 1]), array([0, 0, 2]))
        # Those indices are obtained back using np.unravel_index(valid_indices, is_valid.shape)
        valid_indices = np.flatnonzero(is_valid)
        # Sample valid transitions that will constitute the minibatch of size batch_size
        sampled_indices = np.random.choice(valid_indices, size=batch_size, replace=True) # , p=self.normalized_diversity[is_valid])
        # Unravel the indexes, i.e. recover the batch and env indices.
        # Example: if sampled_indices = [0, 3, 5], then batch_indices = [0, 1, 1] and env_indices = [0, 0, 2]
        batch_indices, env_indices = np.unravel_index(sampled_indices, is_valid.shape)

        # Split the indexes between real and virtual transitions.
        nb_virtual = int(self.her_ratio * batch_size)
        virtual_batch_indices, real_batch_indices = np.split(batch_indices, [nb_virtual])
        virtual_env_indices, real_env_indices = np.split(env_indices, [nb_virtual])

        # Get real and virtual data
        real_data = self._get_real_samples(real_batch_indices, real_env_indices, env)
        # Create virtual transitions by sampling new desired goals and computing new rewards
        virtual_data = self._get_virtual_samples(virtual_batch_indices, virtual_env_indices, env)

        # Concatenate real and virtual data
        observations = {
            key: th.cat((real_data.observations[key], virtual_data.observations[key]))
            for key in virtual_data.observations.keys()
        }
        actions = th.cat((real_data.actions, virtual_data.actions))
        next_observations = {
            key: th.cat((real_data.next_observations[key], virtual_data.next_observations[key]))
            for key in virtual_data.next_observations.keys()
        }
        dones = th.cat((real_data.dones, virtual_data.dones))
        rewards = th.cat((real_data.rewards, virtual_data.rewards))

        return DictReplayBufferSamples(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            rewards=rewards,
        )
        
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
        
        # reward_weights = self.reward_weights[batch_indices, env_indices].reshape(-1, self.reward_dim)
        reward_weights = np.repeat(np.expand_dims(self.scheduler.get_current_weights(), axis=0), repeats=len(batch_indices), axis=0)
        rewards = self.rewards[batch_indices, env_indices].reshape(-1, self.reward_dim)
        rewards = rewards.reshape(-1, 1, self.reward_dim) @ reward_weights.reshape(-1, 1, self.reward_dim).swapaxes(1, 2)
        rewards = rewards.reshape(-1, 1)
        
        if self.use_uvfa:
            obs_["weights"] = reward_weights
            next_obs_["weights"] = reward_weights

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
            rewards=self.to_torch(self._normalize_reward(rewards, env)),
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
        rewards = rewards[0].astype(np.float32) # env_method returns a list containing one element
        reward_weights = self.scheduler.sample_batch(len(batch_indices))
                
        rewards = rewards.reshape(-1, 1, self.reward_dim) @ reward_weights.reshape(-1, 1, self.reward_dim).swapaxes(1, 2)
        rewards = rewards.reshape(-1, 1)

        if self.use_uvfa:
            obs["weights"] = reward_weights
            next_obs["weights"] = reward_weights
        
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
            rewards=self.to_torch(self._normalize_reward(rewards.reshape(-1, 1), env)),
        )
        
    def __getstate__(self):
        state = self.__dict__.copy()
        del state["scheduler"]
        del state["env"]
        return state
    
    def __setstate__(self, state):
        assert "env" not in state
        assert "scheduler" not in state
        self.__dict__.update(state)
        self.env = None
        
    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
        self.reward_dim = self.scheduler.reward_dim