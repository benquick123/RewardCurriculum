import os
from time import sleep
import multiprocessing as mp


def run_command(command):
    print("running command:", command)
    os.system(command)
    print("end command:", command)


if __name__ == "__main__":
    
    start_seed = 0
    n_seeds = 5
    max_processes = 4
    
    experiment_configs = [
        # "--config_path configs/experiment_sac_single_task.json --env_name LunarLanderContinuous-v2 --environment.wrapper_kwargs.0.reward_threshold 99",
        # "--config_path configs/experiment_sac_single_task_non_sparse.json --env_name LunarLanderContinuous-v2",
        # "--config_path configs/experiment_sac_single_task.json --env_name dm_control/acrobot-swingup-v0",
        # "--config_path configs/experiment_sac_single_task_non_sparse.json --env_name dm_control/acrobot-swingup-v0",
        # "--config_path configs/experiment_sac_single_task.json --env_name dm_control/acrobot-swingup_sparse-v0",
        # "--config_path configs/experiment_sac_single_task_non_sparse.json --env_name dm_control/acrobot-swingup_sparse-v0",
        # "--config_path configs/experiment_sac_single_task.json --env_name dm_control/ball_in_cup-catch-v0",
        # "--config_path configs/experiment_sac_single_task_non_sparse.json --env_name dm_control/ball_in_cup-catch-v0",
        # "--config_path configs/experiment_sac_single_task.json --env_name dm_control/cartpole-balance-v0",
        # "--config_path configs/experiment_sac_single_task_non_sparse.json --env_name dm_control/cartpole-balance-v0",
        # "--config_path configs/experiment_sac_single_task.json --env_name dm_control/cartpole-balance_sparse-v0",
        # "--config_path configs/experiment_sac_single_task_non_sparse.json --env_name dm_control/cartpole-balance_sparse-v0",
        # "--config_path configs/experiment_sac_single_task.json --env_name dm_control/cartpole-swingup-v0",
        # "--config_path configs/experiment_sac_single_task_non_sparse.json --env_name dm_control/cartpole-swingup-v0",
        # "--config_path configs/experiment_sac_single_task.json --env_name dm_control/cartpole-swingup_sparse-v0",
        # "--config_path configs/experiment_sac_single_task_non_sparse.json --env_name dm_control/cartpole-swingup_sparse-v0",
        # "--config_path configs/experiment_sac_single_task.json --env_name dm_control/cheetah-run-v0",
        # "--config_path configs/experiment_sac_single_task_non_sparse.json --env_name dm_control/cheetah-run-v0",
        # "--config_path configs/experiment_sac_single_task.json --env_name dm_control/finger-spin-v0",
        # "--config_path configs/experiment_sac_single_task_non_sparse.json --env_name dm_control/finger-spin-v0",
        # "--config_path configs/experiment_sac_single_task.json --env_name dm_control/finger-turn_easy-v0",
        # "--config_path configs/experiment_sac_single_task_non_sparse.json --env_name dm_control/finger-turn_easy-v0",
        # "--config_path configs/experiment_sac_single_task.json --env_name dm_control/finger-turn_hard-v0",
        # "--config_path configs/experiment_sac_single_task_non_sparse.json --env_name dm_control/finger-turn_hard-v0",
        # "--config_path configs/experiment_sac_single_task.json --env_name dm_control/fish-upright-v0",
        # "--config_path configs/experiment_sac_single_task_non_sparse.json --env_name dm_control/fish-upright-v0",
        # "--config_path configs/experiment_sac_single_task.json --env_name dm_control/fish-swim-v0",
        # "--config_path configs/experiment_sac_single_task_non_sparse.json --env_name dm_control/fish-swim-v0",
        # "--config_path configs/experiment_sac_single_task.json --env_name dm_control/hopper-stand-v0",
        # "--config_path configs/experiment_sac_single_task_non_sparse.json --env_name dm_control/hopper-stand-v0",
        # "--config_path configs/experiment_sac_single_task.json --env_name dm_control/hopper-hop-v0",
        # "--config_path configs/experiment_sac_single_task_non_sparse.json --env_name dm_control/hopper-hop-v0",
        # "--config_path configs/experiment_sac_single_task.json --env_name dm_control/humanoid-stand-v0",
        # "--config_path configs/experiment_sac_single_task_non_sparse.json --env_name dm_control/humanoid-stand-v0",
        # "--config_path configs/experiment_sac_single_task.json --env_name dm_control/humanoid-walk-v0",
        # "--config_path configs/experiment_sac_single_task_non_sparse.json --env_name dm_control/humanoid-walk-v0",
        # "--config_path configs/experiment_sac_single_task.json --env_name dm_control/humanoid-run-v0",
        # "--config_path configs/experiment_sac_single_task_non_sparse.json --env_name dm_control/humanoid-run-v0",
        # "--config_path configs/experiment_sac_single_task.json --env_name dm_control/manipulator-bring_ball-v0",
        # "--config_path configs/experiment_sac_single_task_non_sparse.json --env_name dm_control/manipulator-bring_ball-v0",
        # "--config_path configs/experiment_sac_single_task.json --env_name dm_control/pendulum-swingup-v0",
        # "--config_path configs/experiment_sac_single_task_non_sparse.json --env_name dm_control/pendulum-swingup-v0",
        # "--config_path configs/experiment_sac_single_task.json --env_name dm_control/point_mass-easy-v0",
        # "--config_path configs/experiment_sac_single_task_non_sparse.json --env_name dm_control/point_mass-easy-v0",
        # "--config_path configs/experiment_sac_single_task.json --env_name dm_control/reacher-easy-v0",
        # "--config_path configs/experiment_sac_single_task_non_sparse.json --env_name dm_control/reacher-easy-v0",
        # "--config_path configs/experiment_sac_single_task.json --env_name dm_control/reacher-hard-v0",
        # "--config_path configs/experiment_sac_single_task_non_sparse.json --env_name dm_control/reacher-hard-v0",
        # "--config_path configs/experiment_sac_single_task.json --env_name dm_control/swimmer-swimmer6-v0",
        # "--config_path configs/experiment_sac_single_task_non_sparse.json --env_name dm_control/swimmer-swimmer6-v0",
        # "--config_path configs/experiment_sac_single_task.json --env_name dm_control/swimmer-swimmer15-v0",
        # "--config_path configs/experiment_sac_single_task_non_sparse.json --env_name dm_control/swimmer-swimmer15-v0",
        # "--config_path configs/experiment_sac_single_task.json --env_name dm_control/walker-stand-v0",
        # "--config_path configs/experiment_sac_single_task_non_sparse.json --env_name dm_control/walker-stand-v0",
        # "--config_path configs/experiment_sac_single_task.json --env_name dm_control/walker-walk-v0",
        # "--config_path configs/experiment_sac_single_task_non_sparse.json --env_name dm_control/walker-walk-v0",
        # "--config_path configs/experiment_sac_single_task.json --env_name dm_control/walker-run-v0",
        # "--config_path configs/experiment_sac_single_task_non_sparse.json --env_name dm_control/walker-run-v0",
        
        # closest gym only environments in dm_control suite.
        # "--config_path configs/experiment_sac_single_task.json --env_name dm_control/swimmer-swimmer6-v0",
        # "--config_path configs/experiment_sac_single_task.json --env_name dm_control/walker-walk-v0",
        # "--config_path configs/experiment_sac_single_task_non_sparse.json --env_name dm_control/hopper-hop-v0",
        # "--config_path configs/experiment_sac_single_task_non_sparse.json --env_name dm_control/humanoid-walk-v0",
        # "--config_path configs/experiment_sac_single_task_non_sparse.json --env_name dm_control/walker-walk-v0",
        # "--config_path configs/experiment_sac_single_task_non_sparse.json --env_name dm_control/cheetah-run-v0",
        # "--config_path configs/experiment_sac_single_task_non_sparse.json --env_name dm_control/swimmer-swimmer6-v0",
        # "--config_path configs/experiment_sac_single_task.json --env_name dm_control/hopper-hop-v0",
        # "--config_path configs/experiment_sac_single_task.json --env_name dm_control/humanoid-walk-v0",
        # "--config_path configs/experiment_sac_single_task.json --env_name dm_control/cheetah-run-v0",
        
        # alpgmm
        # "--config_path configs/experiment_sac_alpgmm.json --env_name dm_control/cheetah-run-v0",
        # "--config_path configs/experiment_sac_alpgmm.json --env_name dm_control/hopper-hop-v0",
        # "--config_path configs/experiment_sac_alpgmm.json --env_name dm_control/humanoid-walk-v0",
        # "--config_path configs/experiment_sac_alpgmm.json --env_name dm_control/swimmer-swimmer6-v0",
        # "--config_path configs/experiment_sac_alpgmm.json --env_name dm_control/walker-walk-v0",
        
        # PandaStack
        "--config_path configs/experiment_sac_single_task_sparse.json --env_name PandaStack-v3 --environment.wrapper_kwargs.0.reward_threshold -0.1",
        "--config_path configs/experiment_sac_single_task_dense.json --env_name PandaStack-v3 --environment.wrapper_kwargs.0.reward_threshold -0.1"
    ]
    
    command = "python train.py %s --seed %d"
    processes = []
    
    for seed in range(start_seed, start_seed + n_seeds):
        for experiment_config in experiment_configs:
            command_to_run = command % (experiment_config, seed)
            p = mp.Process(target=run_command, args=[command_to_run])
            p.start()
            processes.append(p)
            sleep(1)
            
            if len(processes) == max_processes:
                for p in processes:
                    p.join()
                processes = []
            
            sleep(1)
            
    for p in processes:
        p.join()