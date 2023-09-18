import copy
import json
import os

from utils.configs import load_config


def save_self(config_path, args, config):
    if not config["log"]:
        return 
    
    path = os.path.join(config["log_path"], config["log"])
    os.makedirs(os.path.join(path, "code"))
    ignore = set(open(".gitignore", "r").read().split("\n") + [".git", ".gitignore", "configs"])
    for root, dirs, files in os.walk("."):
        if any([ignore_string in root for ignore_string in ignore]):
            continue
        
        for file in files:
            if any([ignore_string in file for ignore_string in ignore]):
                continue
            os.makedirs(os.path.join(path, "code", root), exist_ok=True)
            os.system("cp %s %s" % (os.path.join(root, file), os.path.join(path, "code", root, file)))
        # print(root, dirs, files)

    def check_recursive(dictionary):
        for k, v in dictionary.items():
            if isinstance(v, dict):
                dictionary[k] = check_recursive(v)
                
            elif isinstance(v, list) or isinstance(v, tuple) or isinstance(v, set):
                v = list(v)
                for i, el in enumerate(v):
                    v[i] = check_recursive({"k": el})["k"]
            
            elif isinstance(v, int) or isinstance(v, float) or isinstance(v, str) or isinstance(v, bool):
                continue
            
            else:
                dictionary[k] = str(v)
        return dictionary
    
    config_dict = check_recursive(copy.deepcopy(config))
    json.dump(config_dict, open(os.path.join(path, "config.json"), "w"), indent=4)
    
    # save the original config
    original_config = load_config(config_path)
    json.dump(original_config, open(os.path.join(path, "config_original.json"), "w"), indent=4)
    
    with open(os.path.join(path, "args.txt"), "w") as f:
        f.write(str(args))
