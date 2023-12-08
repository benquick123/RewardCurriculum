import gymnasium as gym
import panda_gym
from PIL import Image
import numpy as np

if __name__ == "__main__":
    env = gym.make("PandaMultiRewardStackDense-v3")
    observation, info = env.reset()
    ee_goal = observation["desired_goal"][:3]
    obj_goal = [observation["desired_goal"][3:6], observation["desired_goal"][6:]]

    ee_ach = observation["achieved_goal"][:3]
    obj_ach = [observation["achieved_goal"][3:6], observation["achieved_goal"][6:]]

    truncated = False
    gripper = 1
    grasp = False
    obj_in_question = 0
    up = 0
    up_counter = 10
    
    action = np.concatenate((ee_ach - obj_ach[obj_in_question], [gripper]))
    action[2] += up 
    print(action.shape)
    step_no = 0
    while not truncated:
        
        # action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        ee_ach = observation["achieved_goal"][:3]
        obj_ach = [observation["achieved_goal"][3:6], observation["achieved_goal"][6:]]
        action = np.concatenate((obj_ach[obj_in_question] - ee_ach, [gripper]))
        
        if reward[obj_in_question+1] > -0.025:
            gripper = -1
            grasp = True
            up = 0.1
            up_counter = 10
            
        if grasp:
            action = np.concatenate((obj_goal[obj_in_question] - obj_ach[obj_in_question], [gripper]))
            
        if reward[obj_in_question+3] > -0.025:
            gripper = 1
            grasp = False
            obj_in_question = 1
            up = 0.1
            up_counter = 10
            
        if up_counter == 0:
            up = 0
        
        print(reward)
        action[2] += up
        up_counter -= 1
        action *= 2
        
        # env.unwrapped.sim.set_base_pose("panda", env.unwrapped.sim.get_base_position("panda") + (np.random.rand(3) - 0.5) * [0.1, 0.1, 0.0], np.array([0.0, 0.0, 0.0, 0.1]))
        img = Image.fromarray(env.render())
        img.save("tmp.png")
        
        # print(action, observation.shape, reward, terminated, truncated, info, end="")
        input()
        
        step_no += 1
        