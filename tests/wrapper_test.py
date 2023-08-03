import importlib
import sys
sys.path.append("/mnt/d/Benjamin/RewardCurriculum/")

from dm_control import suite

from utils.env_wrappers import get_env

if __name__ == "__main__":
    
    # env = get_env("LunarLanderContinuous-v2", wrappers=["SparseRewardWrapper", "__envwrapper__", "gym.wrappers.FlattenObservation"], wrapper_kwargs=[{"reward_threshold": 99}, {}, {}]) 
    # env.reset()
    # action = env.action_space.sample()
    # observation, reward, terminal, timeout, info = env.step(action)
    # print(observation.shape, reward.shape, action.shape)
    
    for domain, task in suite.BENCHMARKING:
        env_name = "dm_control/" + domain.lower() + "-" + task.lower() + "-v0"
        print(env_name)
        
        env = get_env(env_name, wrappers=["SparseRewardWrapper", "__envwrapper__", "gym.wrappers.FlattenObservation"], wrapper_kwargs=[{"reward_threshold": 0.9}, {}, {}])
        
        env.reset()
        action = env.action_space.sample()
        observation, reward, terminal, timeout, info = env.step(action)
        
        print(env)
        print(f"action: {action.shape}, observation: {observation.shape}, reward: {reward.shape}, info: {info}")
        exit()
        