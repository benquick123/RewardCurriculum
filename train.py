import warnings

def warn(message, category='', stacklevel=1, source=''):
    conditions = [
        category == DeprecationWarning,
        message.startswith("rich is experimental/alpha"),
    ]
    if any(conditions):
        return None
    else:
        print(message, category, stacklevel, source, sep=" | ")

warnings.warn = warn

import os

from stable_baselines3.common.vec_env import SubprocVecEnv

from utils.args import parse_args
from utils.callbacks import SchedulerCallback
from utils.configs import get_config
from utils.logging import save_self
from utils.env_wrappers import get_env, make_vec_env

if __name__ == "__main__":
    args, remaining_args = parse_args()
    config = get_config(args.config_path, args, remaining_args)
    save_self(args.config_path, config)
    
    make_env_fn = lambda wrappers, wrapper_kwargs : get_env(config["environment"]["env_name"], wrappers=wrappers, wrapper_kwargs=wrapper_kwargs)
    
    env = make_vec_env(make_env_fn, 
                       n_envs=config["environment"]["n_envs"], 
                       env_kwargs={"wrappers": config["environment"]["wrappers"], "wrapper_kwargs": config["environment"]["wrapper_kwargs"]},
                       monitor_kwargs={"allow_early_resets": True},
                       seed=config["seed"], vec_env_cls=SubprocVecEnv)
    
    callback = SchedulerCallback(config)
    if config["log"]:
        wrappers = ["SparseRewardWrapper", "gym.wrappers.FlattenObservation"]
        wrapper_kwargs = []
        for i in range(len(wrappers)):
            if wrappers[i] in config["environment"]["wrappers"]:
                wrapper_kwargs.append(config["environment"]["wrapper_kwargs"][config["environment"]["wrappers"].index(wrappers[i])])
            else:
                wrapper_kwargs.append({})
        
        from utils.callbacks import EvalCallback
        eval_env = make_vec_env(make_env_fn, n_envs=1, env_kwargs={"wrappers": wrappers, "wrapper_kwargs": wrapper_kwargs}, seed=config["seed"], vec_env_cls=SubprocVecEnv)
        callback = [EvalCallback(eval_env=eval_env, warn=False, **config["eval_kwargs"]), callback]
    
    learner = config["learner_class"]("MlpPolicy", env, **config["learner_kwargs"])
    learner.learn(callback=callback, **config["train_kwargs"])
    learner.save(os.path.join(config["log_path"], config["log"], "final.zip"))
    