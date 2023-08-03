import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="Path to experiment config")
    parser.add_argument("--env_name", type=str, help="String name of the environment. Must be registered with gym beforehand.")
    parser.add_argument("--seed", type=int, help="Experiment seed")
    args, remaining_args = parser.parse_known_args()
    return args, remaining_args
