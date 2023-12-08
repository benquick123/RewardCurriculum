import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="", help="Path to experiment config.")
    parser.add_argument("--env_name", type=str, help="String name of the environment. Must be registered with gym beforehand.")
    parser.add_argument("--seed", type=int, help="Experiment seed")
    parser.add_argument("--continue_from", type=str, default=None, help="Path to the log folder to continue training from.")
    parser.add_argument("--continue_mode", type=str, default="final", help="How to continue training; using last or best previous model.")
    args, remaining_args = parser.parse_known_args()
    return args, remaining_args
