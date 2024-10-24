from typing import Any, Dict

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance


class MultiRewardPickAndPlaceObstacle(Task):
    
    def __init__(
        self,
        sim: PyBullet,
        get_ee_position,
        reward_type: str = "sparse",
        distance_threshold: float = 0.05,
        goal_xy_range: float = 0.3,
        goal_z_range: float = 0.2,
        obj_xy_range: float = 0.3,
        ee_xy_range: float = 0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.04
        self.obstacle_thickness = 0.02
        self.get_ee_position = get_ee_position
        
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([-self.obstacle_thickness, goal_xy_range / 2, goal_z_range])
        self.ee_goal_range_low = np.array([-ee_xy_range / 2, -ee_xy_range / 2, 0])
        self.ee_goal_range_high = np.array([ee_xy_range / 2, ee_xy_range / 2, ee_xy_range])
        self.obj_range_low = np.array([self.obstacle_thickness, -obj_xy_range, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_box(
            body_name="object",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )
        self.sim.create_box(
            body_name="target",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, 0.05]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )
        self.sim.create_sphere(
            body_name="ee_target",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )
        self.sim.create_box(
            body_name="obstacle",
            half_extents=np.array([self.obstacle_thickness / 2, (self.goal_range_high[1] - self.goal_range_low[1]), 0.05]),
            mass=0.0,
            position=np.array([0.0, 0.0, 0.1 / 2]),
            rgba_color=np.array([0.95, 0.95, 0.95, 1]),
        )

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        object_position = self.sim.get_base_position("object")
        object_rotation = self.sim.get_base_rotation("object")
        object_velocity = self.sim.get_base_velocity("object")
        object_angular_velocity = self.sim.get_base_angular_velocity("object")
        observation = np.concatenate([object_position, object_rotation, object_velocity, object_angular_velocity])
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        object_position = np.array(self.sim.get_base_position("object"))
        ee_position = self.get_ee_position()
        return np.concatenate([ee_position, object_position])

    def reset(self) -> None:
        self.goal = self._sample_goal()
        object_position = self._sample_object()
        
        # set the ee goal to the object position
        self.goal[:3] = object_position
                
        self.sim.set_base_pose("ee_target", self.goal[:3], np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("target", self.goal[3:], np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0]))
        
        rewards = self.compute_reward(self.get_achieved_goal(), self.goal, {})
        if self.reward_type == "dense":
            rewards = (rewards > -self.distance_threshold).astype(float)
        
        if np.any(rewards):
            self.reset()

    def _sample_goal(self) -> np.ndarray:
        """Sample a goal."""
        goal = np.array([0.0, 0.0, self.object_size / 2])  # z offset for the cube center
        
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        if self.np_random.random() < 0.3:
            noise[2] = 0.0
        goal += noise
        
        ee_goal = self.np_random.uniform(self.ee_goal_range_low, self.ee_goal_range_high)
        return np.concatenate([ee_goal, goal])

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = np.array([0.0, 0.0, self.object_size / 2])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        return object_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal[3:], desired_goal[3:])
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        
        assert achieved_goal.shape == desired_goal.shape
        was_1d = False
        if len(achieved_goal.shape) == 1:
            achieved_goal = achieved_goal.reshape(1, -1)
            desired_goal = desired_goal.reshape(1, -1)
            was_1d = True
            
        ee_position = achieved_goal[:, :3]
        object_position = achieved_goal[:, 3:]
        ee_desired = desired_goal[:, :3]
        object_desired = desired_goal[:, 3:]
        
        rewards = np.stack([
            # ee at position
            -distance(ee_position, ee_desired),
            # holding object1 (multiplied by 2 to offset for smaller threshold)
            -distance(ee_position, object_position) * 2,
            # MAIN REWARD
            # object1 at position
            -distance(object_position, object_desired)
        ], axis=-1)
            
        if was_1d:
            rewards = rewards.reshape(-1)
            
        if self.reward_type == "sparse":
            rewards = (rewards > -self.distance_threshold).astype(float)
        return rewards
