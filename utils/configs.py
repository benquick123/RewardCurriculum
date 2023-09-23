import json
import numpy as np
import os
from datetime import datetime
import importlib


def config_merge(config_main, config_other):
    assert isinstance(config_main, dict)
    assert isinstance(config_other, dict)
    
    all_keys = set(config_main.keys()).union(config_other.keys())
    config = {}
    for key in all_keys:
        if key in config_other:
            config[key] = config_other[key]
        
        if key in config_main:
            config[key] = config_main[key]
            
        if isinstance(config[key], dict) and key in config_main and key in config_other:
            config[key] = config_merge(config_main[key], config_other[key])
            
    return config


def load_config(path):
    config_main = json.load(open(path, "r"))
    if "template" in config_main:
        if isinstance(config_main["template"], str):
            config_main["template"] = [config_main["template"]]
        
        config = dict(config_main)
        for template in config_main["template"]:
            config_template = load_config(template)
            config = config_merge(config, config_template)
            
        del config["template"]
        return config
    else:
        return config_main


def get_config(path, args, remaining_args):
    config = load_config(path)
    
    # go through remaining args
    for i in range(0, len(remaining_args), 2):
        arg = remaining_args[i].replace("--", "").split(".")
        value = remaining_args[i+1]
        current = config
        for _arg in arg[:-1]:            
            if _arg.isdigit():
                _arg = int(_arg)
                assert len(current) > _arg, "Provided index is out of range."
            elif _arg not in current:
                current[_arg] = dict()
                
            current = current[_arg]
        
        # take care of some basic type conversions
        try:
            value = eval(value)
        except:
            print(f"Couldn't parse remaining arg `{remaining_args[i]}` ({value}). Will keep as str.")        
        current[arg[-1]] = value
    
    # environment hyperparams
    config["environment"]["env_name"] = args.env_name
    config["environment"]["n_envs"] = min(config["learner_kwargs"]["train_freq"][0], config["environment"].get("n_envs", float("inf")))        

    ##### seeds ####
    if args.seed is not None:
        seed = args.seed
    elif "seed" in config:
        seed = config["seed"]
    else:
        seed = np.random.randint(0, 99999999)
    
    config["seed"] = seed
    config["learner_kwargs"]["seed"] = seed + 1
    config["learner_kwargs"]["scheduler_kwargs"]["seed"] = seed + 2
    
    ##### number of steps vs. buffer_size #####
    if "buffer_size" in config["learner_kwargs"]:
        config["learner_kwargs"]["buffer_size"] = min(config["learner_kwargs"]["buffer_size"], config["train_kwargs"]["total_timesteps"])
    else:
        config["learner_kwargs"]["buffer_size"] = config["train_kwargs"]["total_timesteps"]
    
    ##### logging #####
    if config["log"]:
        ##### construct log path #####
        if isinstance(config["log"], bool):
            config["log"] = str(datetime.now()).split(".")[0].replace(":", "-").replace(" ", "_")
        
        if "log_path" not in config:
            config["log_path"] = "."
        
        experiment_id = args.env_name.split("/")[-1] + "_" + config["learner_class"].split(".")[-1].lower() + "_" + config["learner_kwargs"]["scheduler_class"].split(".")[-1].lower() + "_" + str(config["seed"])
        config["log"] += "_" + experiment_id
        if len(remaining_args) > 0:
            config["log"] += "_" + "_".join([arg_name[2:].split(".")[-1] + "=" + value for arg_name, value in zip(remaining_args[0::2], remaining_args[1::2])])
        # os.makedirs(os.path.join(config["log_path"], config["log"]), exist_ok=exist_ok)
        
        config["tb_log_name"] = experiment_id
        config["learner_kwargs"]["tensorboard_log"] = os.path.join(config["log_path"], config["log"], "tb")
        
        ##### deal with evaluation #####
        if "eval_kwargs" not in config:
            config["eval_kwargs"] = dict()
        config["eval_kwargs"]["log_path"] = os.path.join(config["log_path"], config["log"], "evaluations")
        config["eval_kwargs"]["best_model_save_path"] = os.path.join(config["log_path"], config["log"], "evaluations")
        if "log_model_interval" not in config:
            config["log_model_interval"] = 50000
    
    ##### learner cls #####
    if "learner_class" in config:
        module_split = config["learner_class"].split(".")
        module_path = "/".join(module_split[:-1]) + ".py"
        spec = importlib.util.spec_from_file_location("", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config["learner_class"] = module.__dict__[module_split[-1]]
    
    if "train_freq" in config["learner_kwargs"] and isinstance(config["learner_kwargs"]["train_freq"], list):
        config["learner_kwargs"]["train_freq"] = tuple(config["learner_kwargs"]["train_freq"])
        assert config["learner_kwargs"]["train_freq"][1] == "episode", "Update schedules other than 'episode' are not supported."
    
    ##### scheduler cls #####
    if "scheduler_class" in config["learner_kwargs"]:
        module_split = config["learner_kwargs"]["scheduler_class"].split(".")
        module_path = "/".join(module_split[:-1]) + ".py"
        spec = importlib.util.spec_from_file_location("", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config["learner_kwargs"]["scheduler_class"] = module.__dict__[module_split[-1]]
        
    config["learner_kwargs"]["scheduler_kwargs"]["update_frequency"] = min(config["learner_kwargs"]["train_freq"][0], 
                                                                           config["learner_kwargs"]["scheduler_kwargs"].get("update_frequency", float("inf")))
        
    # if "scheduler_kwargs" not in config["learner_kwargs"]:
    #     config["learner_kwargs"]["scheduler_kwargs"] = dict()
    # config["learner_kwargs"]["scheduler_kwargs"]["log_path"] = os.path.join(config["log_path"], config["log"], "reward_weights.csv")
            
    return config