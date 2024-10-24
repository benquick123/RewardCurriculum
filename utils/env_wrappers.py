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

class DictObservationWrapper(gym.Wrapper):
    
    def __init__(self, env: Env, observation_keys: list[str] = [], observation_dims: list[tuple] = []):
        super().__init__(env)
        
        self.observation_keys = observation_keys
        self.observations_dims = observation_dims
        
        observation_space = {"observation": self.env.observation_space}
        for key, dim in zip(self.observation_keys, self.observations_dims):
            observation_space[key] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=dim, dtype=np.float32)
        self.observation_space = gym.spaces.Dict(observation_space)
        
    def reset(self, **kwargs):
        observation, reset_info = self.env.reset(**kwargs)
        return self.observation(observation), reset_info
    
    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(observation), reward, terminated, truncated, info
    
    def observation(self, observation):
        
        observation = {"observation": observation}
        for key, dim in zip(self.observation_keys, self.observations_dims):
            observation[key] = np.zeros(dim) * np.nan
        return observation


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
    

class DelayedRewardWrapper(gym.RewardWrapper):
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        self.episode_reward = 0
        return super().reset(seed=seed, options=options)
    
    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        info["original_reward"] = reward
        self.episode_reward += reward
        
        if terminated:
            reward = self.episode_reward
        else:
            reward = 0
        return observation, self.reward(reward), terminated, truncated, info
    
    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        return reward
    

class AuxRewardWrapper(gym.RewardWrapper):
    
    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, self.reward(reward, observation, action, info), terminated, truncated, info
    
    def reward(self, reward, observation, action, info=None):
        return np.array([reward], dtype=float)
    
    def compute_reward(self, achieved_goal, desired_goal, infos):
        return self.env.compute_reward(achieved_goal, desired_goal, infos).reshape(len(achieved_goal), -1)


class PotentialRewardWrapper(gym.RewardWrapper):
    
    def __init__(self, env: Env):
        self._previous_reward = None
        super().__init__(env)
        

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


# gymnasium environment wrappers
class AntAuxRewardWrapper(AuxRewardWrapper):
            
    def reward(self, reward, observation, action, info):
        return np.array([
            info["reward_survive"],
            -info["reward_ctrl"],
            info["reward_forward"],
            # reward
        ])
    

class HalfcheetahAuxRewardWrapper(AuxRewardWrapper):
    
    def reward(self, reward, observation, action, info):
        return np.array([
            info["reward_run"],
            info["reward_ctrl"],
            reward
        ])


class HopperAuxRewardWrapper(AuxRewardWrapper):
    
    def reward(self, reward, observation, action, info):
        return np.array([
            self.env.unwrapped.healthy_reward,
            self.env.unwrapped._forward_reward_weight * info["x_velocity"],
            -self.env.unwrapped.control_cost(action),
            reward
        ])
    

class HumanoidAuxRewardWrapper(AuxRewardWrapper):
    
    def reward(self, reward, observation, action, info):
        return np.array([
            info["reward_survive"],
            info["reward_linvel"],
            info["reward_quadctrl"],
            reward
        ])
    
    
class SwimmerAuxRewardWrapper(AuxRewardWrapper):
    
    def reward(self, reward, observation, action, info):
        return np.array([
            info["reward_fwd"],
            info["reward_ctrl"],
            reward
        ])


class Walker2DAuxRewardWrapper(AuxRewardWrapper):
    
    def reward(self, reward, observation, action, info):
        return np.array([
            self.env.unwrapped.healthy_reward,
            self.env.unwrapped._forward_reward_weight * info["x_velocity"],
            -self.env.unwrapped.control_cost(action),
            reward
        ])



# dm_control environment wrappers

class AcrobotAuxRewardWrapper(AuxRewardWrapper):
    
    def reward(self, reward, observation, action):
        joint_angles = np.rad2deg(np.angle(observation["orientations"][2:] + 1j * observation["orientations"][:2]))
        # 0 degrees is up:
        joint_angles -= 90
        # make it between 0 and 360
        joint_angles %= 360
        
        return np.array([
            # velocities in various directions
            observation["velocity"][0] < 1.0,
            observation["velocity"][0] > 2.0,
            observation["velocity"][1] < 1.0,
            observation["velocity"][1] > 2.0,
            # pole angles in specific intervals
            joint_angles[0] > 30 and joint_angles[0] < 60,
            joint_angles[0] > 300 and joint_angles[1] < 330,
            joint_angles[0] > 165 and joint_angles[0] < 195,
            joint_angles[1] > 345 and joint_angles[1] < 15,
            joint_angles[1] > 30 and joint_angles[1] < 60,
            joint_angles[1] > 300 and joint_angles[1] < 330,
            reward
        ], dtype=float)
    
    
class BallInCupAuxRewardWrapper(AuxRewardWrapper):
    
    def reward(self, reward, observation, action):
        return np.array([
            np.abs(observation["velocity"][0]) > 1.5,
            np.abs(observation["velocity"][1]) > 1.5,
            np.abs(observation["velocity"][2]) > 1.5,
            np.abs(observation["velocity"][3]) > 1.5,
            np.abs(observation["velocity"][2]) < 0.5,
            np.abs(observation["velocity"][3]) < 0.5,
            observation["position"][2] < -0.2, # left of the cup
            observation["position"][2] > 0.2, # right of the cup
            np.abs(observation["position"][2]) < 0.1, # under the cup
            observation["position"][3] > 0.24, # higher than the cup
            reward
        ], dtype=float)
    
    
class CartpoleAuxRewardWrapper(AuxRewardWrapper):
    
    def reward(self, reward, observation, action):
        pole_angle = np.rad2deg(np.angle(observation["position"][2] + 1j * observation["position"][1]))
        # 0 degrees is up:
        pole_angle -= 90
        # make it between 0 and 360
        pole_angle %= 360
        
        return np.array([
            observation["velocity"][0] < -0.5,
            observation["velocity"][0] > 0.5,
            observation["velocity"][1] < -0.5,
            observation["velocity"][1] > 0.5,
            observation["position"][0] < -0.5,
            observation["position"][0] > 0.5,
            pole_angle > 15 and pole_angle < 97.5,
            pole_angle > 97.5 and pole_angle < 180,
            pole_angle > 180 and pole_angle < 262.5,
            pole_angle > 262.5 and pole_angle < 345,
            reward
        ], dtype=float)


class CheetahAuxRewardWrapper(AuxRewardWrapper):
    
    def reward(self, reward, observation, action):
        return np.array([
            observation["velocity"][0] > 0.5,
            observation["velocity"][0] < -0.5,
            observation["velocity"][1] > 0.5,
            np.abs(observation["velocity"][3]) > 1.0,
            np.abs(observation["velocity"][6]) > 1.0,
            np.sign(observation["velocity"][3]) == np.sign(observation["velocity"][6]),
            np.sign(observation["velocity"][3]) != np.sign(observation["velocity"][6]),
            np.all(np.sign(observation["velocity"][3:6]) == -1),
            np.all(np.sign(observation["velocity"][3:6]) == 1),
            np.all(np.sign(observation["velocity"][6:9]) == 1),
            reward
        ], dtype=float)
    
    
class FingerAuxRewardWrapper(AuxRewardWrapper):
    
    def reward(self, reward, observation, action):
        return np.array([
            observation["touch"][0] > 1.0,
            observation["touch"][1] > 1.0,
            np.abs(observation["velocity"][0]) > 0.5,
            np.abs(observation["velocity"][1]) > 0.5,
            observation["velocity"][2] > 0.1,
            observation["velocity"][2] < -0.1,
            observation["position"][0] < 0.1 and observation["position"][0] > -0.1,
            observation["position"][1] < 0.1 and observation["position"][1] > -0.1,
            observation["position"][2] > 0,
            observation["position"][2] < 0,
            reward
        ], dtype=float)
    
    
class FishAuxRewardWrapper(AuxRewardWrapper):
    
    def reward(self, reward, observation, action):
        return np.array([
            # whether fish is orientated in various directions
            observation["upright"] > 0.9,
            observation["upright"] < -0.9,
            np.abs(observation["upright"]) < 0.1,
            # tail rotations
            np.sign(observation["velocity"][0]) == np.sign(observation["velocity"][2]),
            np.sign(observation["velocity"][0]) != np.sign(observation["velocity"][2]),
            # fin rotations
            np.abs(observation["joint_angles"][3]) > 0.2,
            np.abs(observation["joint_angles"][5]) > 0.2,
            np.sign(observation["velocity"][3]) == np.sign(observation["velocity"][5]),
            np.sign(observation["velocity"][3]) != np.sign(observation["velocity"][5]),
            # fins facing forward or backward
            np.sum(np.abs(observation["joint_angles"][[4, 6]])) > 1.0,
            reward
        ], dtype=float)
    

"""
class HopperAuxRewardWrapper(AuxRewardWrapper):
    
    def reward(self, reward, observation, action):
        # reward joint velocities going in the opposite directions
        are_same = np.all(np.array([np.sign(observation["velocity"][4]), 
                                    np.sign(-observation["velocity"][5]), 
                                    np.sign(observation["velocity"][6])]) == np.sign(observation["velocity"][4]))
        
        return np.array([
            observation["touch"][0] > 1,
            observation["touch"][1] > 1,
            observation["velocity"][0] > 0.5, # x-axis root movement
            observation["velocity"][0] < -0.5, # x-axis root movement
            observation["velocity"][1] > 0.5, # z-axis root movement
            observation["velocity"][1] < -0.5, # z-axis root movement
            observation["velocity"][2] > 0.5, # rotational-axis root movement
            observation["velocity"][2] < -0.5, # rotational-axis root movement
            are_same and np.sign(np.sign(observation["velocity"][4])) == 1,
            are_same and np.sign(np.sign(observation["velocity"][4])) == -1,
            reward
        ], dtype=float)
""" 


"""
class HumanoidAuxRewardWrapper(AuxRewardWrapper):
    
    def reward(self, reward, observation, action):
        return np.array([
            observation["head_height"] > 1.0,
            np.linalg.norm(observation["com_velocity"][:2]) < 1.0,
            np.linalg.norm(observation["com_velocity"][:2]) > 1.0,
            np.abs(np.sum(observation["torso_vertical"])) < 3 * 0.9,
            np.abs(observation["torso_vertical"][0]) > 0.9,
            np.abs(observation["torso_vertical"][1]) > 0.9,
            np.abs(observation["torso_vertical"][2]) > 0.9,
            np.sum(np.abs(observation["joint_angles"])) > len(observation["joint_angles"]) / 2,
            np.sum(np.abs(observation["velocity"])) < len(observation["velocity"]) / 2,
            np.sum(np.abs(observation["velocity"])) > len(observation["velocity"]),
            reward
        ], dtype=float)
"""


class ManipulatorAuxRewardWrapper(AuxRewardWrapper):
    
    def reward(self, reward, observation, action):
        gripper_angles = np.rad2deg(np.angle(observation["arm_pos"][[4, 6], 1] + 1j * observation["arm_pos"][[4, 6], 0]))
        
        return np.array([
            observation["touch"][0] > 0,
            np.any(observation["touch"][1:3] > 0),
            observation["touch"][3] > 0,
            observation["touch"][4] > 0,
            np.linalg.norm(observation["object_pos"][:2] - observation["hand_pos"][:2]) < 0.05,
            np.sum(gripper_angles) > 30, # closed gripper
            np.sum(gripper_angles) < 0, # opened gripper
            observation["object_vel"][0] < -0.2,
            observation["object_vel"][0] > 0.2,
            observation["object_vel"][1] < -0.2,
            reward
        ], dtype=float)
    
    
class PendulumAuxRewardWrapper(AuxRewardWrapper):
    
    def reward(self, reward, observation, action):
        pendulum_angle = np.rad2deg(np.angle(observation["orientation"][1] + 1j * observation["orientation"][0]))
        # pendulum facing up is at 0 deg
        pendulum_angle -= 90
        # degrees must be between 0 and 360
        pendulum_angle %= 360

        return np.array([
            observation["velocity"][0] > 0.5,
            observation["velocity"][0] < -0.5,
            pendulum_angle > 337.5 or pendulum_angle < 22.5,
            pendulum_angle > 22.5 and pendulum_angle < 67.5,
            pendulum_angle > 67.5 and pendulum_angle < 112.5,
            pendulum_angle > 112.5 and pendulum_angle < 157.5,
            pendulum_angle > 157.5 and pendulum_angle < 202.5,
            pendulum_angle > 202.5 and pendulum_angle < 247.5,
            pendulum_angle > 247.5 and pendulum_angle < 292.5,
            pendulum_angle > 292.5 and pendulum_angle < 337.5,
            reward
        ], dtype=float)
    
    
class PointMassAuxRewardWrapper(AuxRewardWrapper):
    
    def reward(self, reward, observation, action):
        return np.array([
            observation["velocity"][0] > 0.5,
            observation["velocity"][0] < -0.5,
            observation["velocity"][1] > 0.5,
            observation["velocity"][1] < -0.5,
            observation["position"][0] > 0,
            observation["position"][0] < 0,
            observation["position"][1] > 0,
            observation["position"][1] < 0,
            np.linalg.norm(observation["position"]) > 0.2,
            np.linalg.norm(observation["position"]) < 0.2,
            reward
        ], dtype=float)
    
    
class ReacherAuxRewardWrapper(AuxRewardWrapper):
    
    def reward(self, reward, observation, action):
        return np.array([
            np.linalg.norm(observation["to_target"]) > 0.5,
            np.linalg.norm(observation["to_target"]) < 0.5,
            observation["velocity"][0] > 0.5,
            observation["velocity"][0] < -0.5,
            observation["velocity"][0] > -0.5 and observation["velocity"][0] < 0.5,
            observation["velocity"][1] > 0.5,
            observation["velocity"][1] < -0.5,
            observation["velocity"][1] > -0.5 and observation["velocity"][1] < 0.5,
            np.sign(observation["velocity"][0]) == np.sign(observation["velocity"][1]),
            np.sign(observation["velocity"][0]) != np.sign(observation["velocity"][1]),
            reward
        ], dtype=float)
    
    
"""
class SwimmerAuxRewardWrapper(AuxRewardWrapper):
    
    def reward(self, reward, observation, action):
        body_size = len(observation["joints"])
        return np.array([
            np.linalg.norm(observation["to_target"]) > 0.5,
            np.linalg.norm(observation["to_target"]) < 0.5,
            np.all(np.sign(observation["joints"]) == -1),
            np.all(np.sign(observation["joints"]) == 1),
            np.all(np.sign(observation["joints"][:body_size // 2]) == -1) and np.all(np.sign(observation["joints"][body_size // 2:]) == 1),
            np.all(np.sign(observation["joints"][:body_size // 2]) == 1) and np.all(np.sign(observation["joints"][body_size // 2:]) == -1),
            observation["body_velocities"][0] > 0,
            observation["body_velocities"][0] < 0,
            observation["body_velocities"][1] > 0,
            observation["body_velocities"][1] < 0,
            reward
        ], dtype=float)
"""    
   

"""  
class WalkerAuxRewardWrapper(AuxRewardWrapper):
    
    def reward(self, reward, observation, action):
        orientations = np.rad2deg(np.angle(observation["orientations"][1::2] + 1j * observation["orientations"][::2]))
        orientations -= 90
        orientations %= 360
        
        # reward joint velocities going in the opposite directions
        are_same = np.all(np.array([np.sign(observation["velocity"][3]), 
                                    np.sign(-observation["velocity"][4]), 
                                    np.sign(observation["velocity"][6]), 
                                    np.sign(-observation["velocity"][7])]) == np.sign(observation["velocity"][3]))
        
        return np.array([
            observation["height"] > 1.0,
            observation["height"] < 1.0,
            orientations[0] > 345 or orientations[0] < 15,
            orientations[1] > 30 and orientations[1] < 60 and orientations[4] > 30 and orientations[4] < 60,
            orientations[1] > 30 and orientations[1] < 60 and orientations[4] > 30 and orientations[4] < 60,
            observation["velocity"][0] > 0.5,
            observation["velocity"][0] < -0.5,
            np.abs(observation["velocity"][1]) > 0.5,
            np.sign(observation["velocity"][3]) == np.sign(observation["velocity"][6]),
            are_same,
            reward
        ], dtype=float)
"""
    

# gym environment wrappers
 
class GymLunarlandercontinuousAuxRewardWrapper(AuxRewardWrapper):
    
    def reward(self, reward, observation, action):
        stop_horizontal = (-0.1 < observation[2]) & (observation[2] < 0.1)
        stop_vertical = (-0.1 < observation[3]) & (observation[3] < 0.1)
        stop_rotation = (-0.1 < observation[5]) & (observation[5] < 0.1)
        
        return np.array([
            observation[6],
            observation[7],
            observation[0] < -0.5,
            observation[5] > 0.1,
            observation[5] < -0.1,
            observation[2] < -0.1,
            observation[2] > 0.1,
            observation[3] > 0.1,
            observation[3] < -0.1,
            stop_horizontal & stop_vertical & stop_rotation,
            reward
        ], dtype=float)


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