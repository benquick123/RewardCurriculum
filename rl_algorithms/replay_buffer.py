from typing import Any, Dict, List, Optional

import numpy as np
import torch as th
import copy
import stable_baselines3
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class HerReplayBuffer(stable_baselines3.her.HerReplayBuffer):
    
    def __init__(self, *args, scheduler=None, use_uvfa=True, **kwargs):
        assert kwargs["n_envs"] == 1
        super().__init__(*args, **kwargs)
        
        self.scheduler = scheduler
        self.use_uvfa = use_uvfa
        self.reward_dim = self.scheduler.reward_dim
        self.rewards = np.zeros((self.buffer_size, self.n_envs, self.reward_dim), dtype=np.float32)
        
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
        next_obs_ = self._normalize_obs({key: np.array(obs[batch_indices, env_indices, :]) for key, obs in self.next_observations.items()}, env)
        
        # reward_weights = self.reward_weights[batch_indices, env_indices].reshape(-1, self.reward_dim)
        reward_weights = np.repeat(np.expand_dims(self.scheduler.get_current_weights(), axis=0), repeats=len(batch_indices), axis=0)
        rewards = self.rewards[batch_indices, env_indices].reshape(-1, self.reward_dim)
        rewards = rewards.reshape(-1, 1, self.reward_dim) @ reward_weights.reshape(-1, 1, self.reward_dim).swapaxes(1, 2)
        rewards = rewards.reshape(-1, 1)
        
        if self.use_uvfa:
            obs_["weights"] = reward_weights
            next_obs_["weights"] = reward_weights
        else:
            obs_["weights"] = np.zeros_like(reward_weights)
            next_obs_["weights"] = np.zeros_like(reward_weights)

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
        else:
            obs["weights"] = np.zeros_like(reward_weights)
            next_obs["weights"] = np.zeros_like(reward_weights)
        
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
       