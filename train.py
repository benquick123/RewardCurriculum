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
from utils.logging import save_self, save_times
from utils.env_wrappers import get_env, make_vec_env
import panda_gym


if __name__ == "__main__":
    args, remaining_args = parse_args()
    config = get_config(args.config_path, args, remaining_args)
    save_self(args.config_path, args, remaining_args, config)
    
    make_env_fn = lambda wrappers, wrapper_kwargs, ignore_keyword="ignore", env_init_kwargs={}: get_env(config["environment"]["env_name"], 
                                                                                     wrappers=wrappers, 
                                                                                     wrapper_kwargs=wrapper_kwargs, 
                                                                                     ignore_keyword=ignore_keyword,
                                                                                     env_init_kwargs=env_init_kwargs)
    env = make_vec_env(make_env_fn, 
                       n_envs=config["environment"]["n_envs"], 
                       env_kwargs={"wrappers": config["environment"]["wrappers"], 
                                   "wrapper_kwargs": config["environment"]["wrapper_kwargs"]},
                       monitor_kwargs={"allow_early_resets": True},
                       seed=config["seed"], vec_env_cls=SubprocVecEnv)
    
    config["learner_kwargs"]["reward_dim"] = env.get_attr("reward_dim", indices=[0])[0]
    
    callback = SchedulerCallback(config)
    if config["log"]:        
        wrappers = config["environment"]["wrappers"].copy()
        wrapper_kwargs = config["environment"]["wrapper_kwargs"].copy()
        
        wrappers += ["SingleTaskRewardWrapper"]
        wrapper_kwargs += [{}]
        
        from utils.callbacks import EvalCallback
        try:
            # in the case of stacking, we don't want to have the cube placed on the table (p_low is set and is 0.0)
            eval_env = make_vec_env(make_env_fn, n_envs=1, env_kwargs={"wrappers": wrappers, "wrapper_kwargs": wrapper_kwargs, "ignore_keyword": None, "env_init_kwargs": {"p_low": 0.0}}, seed=config["seed"], vec_env_cls=SubprocVecEnv)
        except ConnectionResetError:
            # in all other cases, we are fine with the default environment
            print("The environment constructor does not support p_low.")
            eval_env = make_vec_env(make_env_fn, n_envs=1, env_kwargs={"wrappers": wrappers, "wrapper_kwargs": wrapper_kwargs, "ignore_keyword": None}, seed=config["seed"], vec_env_cls=SubprocVecEnv)
            
        callback = [EvalCallback(eval_env=eval_env, warn=False, **config["eval_kwargs"]), callback]

    learner = config["learner_class"]("MultiInputPolicy", env, **config["learner_kwargs"])
    if args.continue_from is not None:
        if args.continue_mode == "final":
            param_load_path = os.path.join(args.continue_from, "final.zip")
        elif args.continue_from == "best":
            param_load_path = os.path.join(args.continue_from, "evaluations", "best_model.zip")
        
        # learner.set_parameters(param_load_path)
        learner.load_replay_buffer(os.path.join(args.continue_from, "replay_buffer.pkl"))
        learner.replay_buffer.set_scheduler(learner.scheduler)
    
    try:
        learner.learn(callback=callback, **config["train_kwargs"])
    except KeyboardInterrupt:
        print("Keyboard interrupt.")
    finally:
        times_dict = {
            "simulation_time": learner.simulation_time - learner.cl_train_time - learner.evaluation_time,
            "rl_train_time": learner.rl_train_time,
            "cl_train_time": learner.cl_train_time,
            "evaluation_time": learner.evaluation_time
        }
        print(config["learner_kwargs"]["scheduler_class"], "\n", times_dict)
        
        if config["log"]:
            save_times(times_dict, os.path.join(config["log_path"], config["log"], "times.json"))
            
            learner.save(os.path.join(config["log_path"], config["log"], "final.zip"))
            learner.save_replay_buffer(os.path.join(config["log_path"], config["log"], "replay_buffer.pkl"))
    