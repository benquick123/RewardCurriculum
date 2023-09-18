import gymnasium as gym
import panda_gym
from PIL import Image
import numpy as np

if __name__ == "__main__":
    env = gym.make("PandaStack-v3")
    env = gym.wrappers.FlattenObservation(env)
    env.reset()
    
    print(dir(env.unwrapped.sim.physics_client))
    exit()
    
    truncated = False
    action = np.ones_like(env.action_space.sample()) * 0
    step_no = 0
    while not truncated:
        
        # action = env.action_space.sample()
        action = action * -1
        observation, reward, terminated, truncated, info = env.step(action)
        
        # env.unwrapped.sim.set_base_pose("panda", env.unwrapped.sim.get_base_position("panda") + (np.random.rand(3) - 0.5) * [0.1, 0.1, 0.0], np.array([0.0, 0.0, 0.0, 0.1]))
        img = Image.fromarray(env.render())
        img.save("tmp.png")
        
        print(action, observation.shape, reward, terminated, truncated, info, end="")
        input()
        
        step_no += 1
        