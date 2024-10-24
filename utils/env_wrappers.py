import os
import time
from typing import (Any, Callable, Dict, Optional, SupportsFloat, Tuple, Type,
                    Union)

import gymnasium as gym
import numpy as np
import panda_gym
import stable_baselines3
from gymnasium.core import ActType, Env, ObsType
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.vec_env.patch_gym import _patch_env


class SparseRewardWrapper(gym.RewardWrapper):
    
    def __init__(self, env, reward_threshold, threshold_relationship="larger", **kwargs):
        super(SparseRewardWrapper, self).__init__(env)
        self.reward_threshold = reward_threshold
        self.threshold_relationship = threshold_relationship
        assert self.threshold_relationship in {"larger", "smaller"}, 'self.threshold_relationship not in {"larger", "smaller"}'
        
    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        info["original_reward"] = reward
        return observation, self.reward(reward), terminated, truncated, info
    
    def reward(self, reward):
        # assert isinstance(reward, float) or isinstance(reward, int), f"A dense reward must be a number before converting it to sparse. Got {reward}"
        if self.threshold_relationship == "larger":
            reward = reward >= self.reward_threshold
        else:
            reward = reward <= self.reward_threshold
            
        if isinstance(reward, np.ndarray):
            return reward.astype(float)
        else:
            return float(reward)
    
    def compute_reward(self, achieved_goal, desired_goal, infos):
        reward = self.env.compute_reward(achieved_goal, desired_goal, infos)
        if self.threshold_relationship == "larger":
            reward = reward >= self.reward_threshold
        else:
            reward = reward <= self.reward_threshold
        return reward.astype(float)


class SingleTaskRewardWrapper(gym.RewardWrapper):
    
    def __init__(self, env: Env, task_index=-1):
        self.task_index = task_index
        super().__init__(env)
    
    def reward(self, reward):
        # Only returns the last element from the rewards array, assuming that corresponds to the main reward.
        assert isinstance(reward, np.ndarray), f"Did not receive a numpy array. Got {reward}"
        assert len(reward.shape) == 1
        
        return reward[self.task_index]
    
    def compute_reward(self, achieved_goal, desired_goal, infos):
        return np.expand_dims(self.env.compute_reward(achieved_goal, desired_goal, infos).reshape(len(achieved_goal), -1)[:, self.task_index], axis=-1)


# other utils

class Monitor(stable_baselines3.common.monitor.Monitor):

    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.rewards.append(reward)
        if terminated or truncated:
            self.needs_reset = True
            stacked_rewards = np.stack(self.rewards, axis=0)
            ep_rew_separate = np.sum(stacked_rewards, axis=0)
            ep_rew = np.sum(ep_rew_separate)
            ep_len = len(self.rewards)
            ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6), "r_separate": ep_rew_separate}
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            info["episode"] = ep_info
        self.total_steps += 1
        return observation, reward, terminated, truncated, info
    
    
def make_vec_env(
    env_id: Union[str, Callable[..., gym.Env]],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_class: Optional[Callable[[gym.Env], gym.Env]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    env_kwargs = env_kwargs or {}
    vec_env_kwargs = vec_env_kwargs or {}
    monitor_kwargs = monitor_kwargs or {}
    wrapper_kwargs = wrapper_kwargs or {}
    assert vec_env_kwargs is not None  # for mypy

    def make_env(rank: int) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            # For type checker:
            assert monitor_kwargs is not None
            assert wrapper_kwargs is not None
            assert env_kwargs is not None

            if isinstance(env_id, str):
                # if the render mode was not specified, we set it to `rgb_array` as default.
                kwargs = {"render_mode": "rgb_array"}
                kwargs.update(env_kwargs)
                try:
                    env = gym.make(env_id, **kwargs)  # type: ignore[arg-type]
                except TypeError:
                    env = gym.make(env_id, **env_kwargs)
            else:
                env = env_id(**env_kwargs)
                # Patch to support gym 0.21/0.26 and gymnasium
                env = _patch_env(env)

            if seed is not None:
                # Note: here we only seed the action space
                # We will seed the env at the next reset
                env.action_space.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            # Create the monitor folder if needed
            if monitor_path is not None and monitor_dir is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path, **monitor_kwargs)
            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env, **wrapper_kwargs)
            return env

        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    vec_env = vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)
    # Prepare the seeds for the first reset
    vec_env.seed(seed)
    return vec_env

    
def get_env(env_name, 
            wrappers=["SparseRewardWrapper", "__envwrapper__", "gym.wrappers.FlattenObservation"], 
            wrapper_kwargs=[{}, {}, {}], 
            ignore_keyword=None, 
            env_init_kwargs={}):
    # works with classes in this file and with classes that are imported at the beginning.
    env = gym.make(env_name, **env_init_kwargs)
    
    for i, wrapper_name in enumerate(wrappers):
        if wrapper_name == "__envwrapper__":
            wrapper_name = env_name.split("/")[-1].split("-")[0].replace("_", " ").title().replace(" ", "") + "AuxRewardWrapper"
            
        import utils.env_wrappers as reward_wrapper
        for lib in wrapper_name.split("."):
            reward_wrapper = reward_wrapper.__dict__[lib]
            
        if ignore_keyword is not None and wrapper_kwargs[i].get(ignore_keyword, False):
            continue
        
        if len(wrapper_kwargs[i]) > 0:
            env = reward_wrapper(env, **wrapper_kwargs[i])
        else:
            env = reward_wrapper(env)
        
    return env