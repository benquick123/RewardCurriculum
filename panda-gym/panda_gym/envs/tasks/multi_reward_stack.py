from typing import Any, Dict, Tuple

import math
import numpy as np
from datetime import datetime

from panda_gym.envs.core import Task
from panda_gym.utils import distance


class MultiRewardStack(Task):
    
    def __init__(
        self,
        sim,
        get_ee_position,
        get_fingers_width,
        reward_type="sparse",
        distance_threshold=0.05,
        goal_xy_range=0.3,
        obj_xy_range=0.3,
        ee_xy_range=0.3,
        p_low=0.3
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.04
        self.p_low = p_low
        
        self.get_ee_position = get_ee_position
        self.get_fingers_width = get_fingers_width
        
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, 0])
        self.ee_goal_range_low = np.array([-ee_xy_range / 2, -ee_xy_range / 2, 0])
        self.ee_goal_range_high = np.array([ee_xy_range / 2, ee_xy_range / 2, ee_xy_range])
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_box(
            body_name="object1",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=2000.0,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.1, 0.9, 1.0]),
        )
        self.sim.create_box(
            body_name="target1",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, 0.05]),
            rgba_color=np.array([0.1, 0.1, 0.9, 0.3]),
        )
        self.sim.create_box(
            body_name="object2",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.5, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )
        self.sim.create_box(
            body_name="target2",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.5, 0.0, 0.05]),
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

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        object1_position = np.array(self.sim.get_base_position("object1"))
        object1_rotation = np.array(self.sim.get_base_rotation("object1"))
        object1_velocity = np.array(self.sim.get_base_velocity("object1"))
        object1_angular_velocity = np.array(self.sim.get_base_angular_velocity("object1"))
        object2_position = np.array(self.sim.get_base_position("object2"))
        object2_rotation = np.array(self.sim.get_base_rotation("object2"))
        object2_velocity = np.array(self.sim.get_base_velocity("object2"))
        object2_angular_velocity = np.array(self.sim.get_base_angular_velocity("object2"))
        observation = np.concatenate(
            [
                object1_position,
                object1_rotation,
                object1_velocity,
                object1_angular_velocity,
                object2_position,
                object2_rotation,
                object2_velocity,
                object2_angular_velocity,
            ]
        )
        # no reaching task-specific observation
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        # object1_position = self.sim.get_base_position("object1")
        object2_position = self.sim.get_base_position("object2")
        ee_position = self.get_ee_position()
        achieved_goal = np.concatenate((ee_position, object2_position)) # object1_position, object2_position))
        return achieved_goal

    def reset(self) -> None:
        tmp_goal = self._sample_goal()
        self.object1_position, self.object2_position = self._sample_objects()
        
        self.goal = np.zeros(6)
        self.goal[:2] = self.object2_position[:2] # self.np_random.choice([self.object1_position, self.object2_position])
        self.goal[2] = self.object_size / 2
        # self.goal[3:6] = self.np_random.choice([self.object1_position, self.goal[3:6]], p=[1.0, 0.0])
        if self.np_random.random() < self.p_low:
            self.goal[3:5] = tmp_goal[6:8]
            self.goal[5] = self.object_size / 2
        else:
            self.goal[3:5] = self.object1_position[:2]
            self.goal[5] = 1.5 * self.object_size
        
        self.sim.set_base_pose("ee_target", self.goal[:3], np.array([0.0, 0.0, 0.0, 1.0]))
        # self.sim.set_base_pose("target1", self.goal[3:6], np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("target1", self.object1_position, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("target2", self.goal[3:], np.array([0.0, 0.0, 0.0, 1.0]))
        
        self.sim.set_base_pose("object1", self.object1_position, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object2", self.object2_position, np.array([0.0, 0.0, 0.0, 1.0]))
        
        rewards = self.compute_reward(self.get_achieved_goal(), self.get_goal(), {})
        if self.reward_type == "dense":
            rewards = (rewards > -self.distance_threshold).astype(float)
        
        # if rewards[-1]:
        if np.any(rewards):
            self.reset()

    def _sample_goal(self) -> np.ndarray:
        ee_goal = self.np_random.uniform(self.ee_goal_range_low, self.ee_goal_range_high)
        goal1 = np.array([0.0, 0.0, self.object_size / 2])  # z offset for the cube center
        goal2 = np.array([0.0, 0.0, 3 * self.object_size / 2])  # z offset for the cube center
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        goal1 += noise
        goal2 += noise
        
        return np.concatenate((ee_goal, goal1, goal2))

    def _sample_objects(self) -> Tuple[np.ndarray, np.ndarray]:
        # while True:  # make sure that cubes are distant enough
        object1_position = np.array([0.0, 0.0, self.object_size / 2])
        object2_position = np.array([0.0, 0.0, 3 * self.object_size / 2])
        noise1 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        noise2 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object1_position += noise1
        object2_position += noise2
        return object1_position, object2_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        # must be vectorized !!
        d = -distance(achieved_goal[3:], desired_goal[3:]) / 2
        
        # for ensuring object2 is on top of object1
        d += (np.abs(achieved_goal[5] - 1.5 * self.object_size) < 1e-3) - 1
        # for ensuring object2 is not held by the gripper
        d += ((-distance(achieved_goal[:3], achieved_goal[3:]) - np.abs(self.get_fingers_width() - self.object_size)) < -self.distance_threshold) - 1
        
        return np.array(d > -self.distance_threshold, dtype=bool)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        ## rewards ##
        
        assert achieved_goal.shape == desired_goal.shape
        was_1d = False
        if len(achieved_goal.shape) == 1:
            achieved_goal = achieved_goal.reshape(1, -1)
            desired_goal = desired_goal.reshape(1, -1)
            was_1d = True
        
        ee_position = achieved_goal[:, :3]
        # object1_position = achieved_goal[:, 3:6]
        # object2_position = achieved_goal[:, 6:]
        object2_position = achieved_goal[:, 3:]
        ee_desired = desired_goal[:, :3]
        # object1_desired = desired_goal[:, 3:6]
        # object2_desired = desired_goal[:, 6:]
        object2_desired = desired_goal[:, 3:]
        
        rewards = np.stack([
            # ee at position
            -distance(ee_position, ee_desired),
            # holding object1
            # -distance(ee_position, object1_position),
            # holding object2
            -distance(ee_position, object2_position) - np.abs(self.get_fingers_width() - self.object_size),
            # object1 at position
            # -distance(object1_position, object1_desired),
            # object2 at position
            # -distance(object2_position, object2_desired),
            # MAIN REWARD:
            # object2 on top of object1 at position (divided by 2 to offset for positions when having two objects)
            # -distance(achieved_goal[:, 3:], desired_goal[:, 3:]) / 2,
            
            # object2 near the desired position
            -distance(object2_position, object2_desired)
        ], axis=-1)
        
        # additionally make sure that the object 2 is actually placed on top of 1
        rewards[:, -1] += (np.abs(object2_position[:, -1] - 1.5 * self.object_size) < 1e-3) - 1
        # also make sure that the gripper is NOT holding the object 2
        rewards[:, -1] += (rewards[:, 1] < -self.distance_threshold) - 1
        
        if was_1d:
            rewards = rewards.reshape(-1)
        
        if self.reward_type == "sparse":
            rewards = (rewards > -self.distance_threshold).astype(float)
        return rewards