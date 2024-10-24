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
    max_processes = 3
    
    experiment_configs = [
        # Paper experiments:
        
        # Multi-reward Pick-and-place with TQC
        "--config_path configs/experiment_tqc_random.json --env_name PandaMultiRewardPickAndPlaceDense-v3",
        "--config_path configs/experiment_tqc_manual_task.json --env_name PandaMultiRewardPickAndPlaceDense-v3",
        "--config_path configs/experiment_tqc_alpgmm.json --env_name PandaMultiRewardPickAndPlaceDense-v3",
        "--config_path configs/experiment_tqc_currot.json --env_name PandaMultiRewardPickAndPlaceDense-v3",
        "--config_path configs/experiment_tqc_setter_solver.json --env_name PandaMultiRewardPickAndPlaceDense-v3",
        "--config_path configs/experiment_tqc_sacx.json --env_name PandaMultiRewardPickAndPlaceDense-v3",
        
        "--config_path configs/experiment_tqc_random.json --env_name PandaMultiRewardPickAndPlaceSphereDense-v3",
        "--config_path configs/experiment_tqc_manual_task.json --env_name PandaMultiRewardPickAndPlaceSphereDense-v3",
        "--config_path configs/experiment_tqc_alpgmm.json --env_name PandaMultiRewardPickAndPlaceSphereDense-v3",
        "--config_path configs/experiment_tqc_currot.json --env_name PandaMultiRewardPickAndPlaceSphereDense-v3",
        "--config_path configs/experiment_tqc_setter_solver.json --env_name PandaMultiRewardPickAndPlaceSphereDense-v3",
        "--config_path configs/experiment_tqc_sacx.json --env_name PandaMultiRewardPickAndPlaceSphereDense-v3",
        
        "--config_path configs/experiment_tqc_random.json --env_name PandaMultiRewardPickAndPlaceObstacleDense-v3",
        "--config_path configs/experiment_tqc_manual_task.json --env_name PandaMultiRewardPickAndPlaceObstacleDense-v3",
        "--config_path configs/experiment_tqc_alpgmm.json --env_name PandaMultiRewardPickAndPlaceObstacleDense-v3",
        "--config_path configs/experiment_tqc_currot.json --env_name PandaMultiRewardPickAndPlaceObstacleDense-v3",
        "--config_path configs/experiment_tqc_setter_solver.json --env_name PandaMultiRewardPickAndPlaceObstacleDense-v3",
        "--config_path configs/experiment_tqc_sacx.json --env_name PandaMultiRewardPickAndPlaceObstacleDense-v3",
        
        # Multi-reward Stack with TQC
        "--config_path configs/experiment_tqc_random.json --env_name PandaMultiRewardStackDense-v3",
        "--config_path configs/experiment_tqc_manual_task.json --env_name PandaMultiRewardStackDense-v3",
        "--config_path configs/experiment_tqc_alpgmm.json --env_name PandaMultiRewardStackDense-v3",
        "--config_path configs/experiment_tqc_currot.json --env_name PandaMultiRewardStackDense-v3",
        "--config_path configs/experiment_tqc_setter_solver.json --env_name PandaMultiRewardStackDense-v3",
        "--config_path configs/experiment_tqc_sacx.json --env_name PandaMultiRewardStackDense-v3",
        
        # Ablation tests with universal policy and value functions
        "--config_path configs/experiment_tqc_setter_solver.json --env_name PandaMultiRewardPickAndPlaceSphereDense-v3 --learner_kwargs.use_upfa False --learner_kwargs.use_uvfa False",
        "--config_path configs/experiment_tqc_setter_solver.json --env_name PandaMultiRewardPickAndPlaceSphereDense-v3 --learner_kwargs.use_upfa False",
        
        # MUJOCO experiments:
        # "--config_path configs/experiment_tqc_random.json --env_name Ant-v4",
        # "--config_path configs/experiment_tqc_random.json --env_name HalfCheetah-v4 --environment.wrapper_kwargs '[{\"observation_keys\": [\"weights\"], \"observation_dims\": [(3, )]}, {}]' --learner_kwargs.reward_dim 3",
        # "--config_path configs/experiment_tqc_random.json --env_name Hopper-v4",
        # "--config_path configs/experiment_tqc_random.json --env_name Humanoid-v4",
        # "--config_path configs/experiment_tqc_random.json --env_name Swimmer-v4 --environment.wrapper_kwargs '[{\"observation_keys\": [\"weights\"], \"observation_dims\": [(3, )]}, {}]' --learner_kwargs.reward_dim 3",
        # "--config_path configs/experiment_tqc_random.json --env_name Walker2d-v4",
        
        # "--config_path configs/experiment_tqc_manual_task.json --env_name Ant-v4",
        # "--config_path configs/experiment_tqc_manual_task.json --env_name HalfCheetah-v4 --environment.wrapper_kwargs '[{\"observation_keys\": [\"weights\"], \"observation_dims\": [(3, )]}, {}]' --learner_kwargs.reward_dim 3",
        # "--config_path configs/experiment_tqc_manual_task.json --env_name Hopper-v4",
        # "--config_path configs/experiment_tqc_manual_task.json --env_name Humanoid-v4",
        # "--config_path configs/experiment_tqc_manual_task.json --env_name Swimmer-v4 --environment.wrapper_kwargs '[{\"observation_keys\": [\"weights\"], \"observation_dims\": [(3, )]}, {}]' --learner_kwargs.reward_dim 3",
        # "--config_path configs/experiment_tqc_manual_task.json --env_name Walker2d-v4",
        
        # "--config_path configs/experiment_tqc_setter_solver.json --env_name Ant-v4",
        # "--config_path configs/experiment_tqc_setter_solver.json --env_name HalfCheetah-v4 --environment.wrapper_kwargs '[{\"observation_keys\": [\"weights\"], \"observation_dims\": [(3, )]}, {}]' --learner_kwargs.reward_dim 3",
        # "--config_path configs/experiment_tqc_setter_solver.json --env_name Hopper-v4",
        # "--config_path configs/experiment_tqc_setter_solver.json --env_name Humanoid-v4",
        # "--config_path configs/experiment_tqc_setter_solver.json --env_name Swimmer-v4 --environment.wrapper_kwargs '[{\"observation_keys\": [\"weights\"], \"observation_dims\": [(3, )]}, {}]' --learner_kwargs.reward_dim 3",
        # "--config_path configs/experiment_tqc_setter_solver.json --env_name Walker2d-v4",
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
