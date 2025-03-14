import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from time import time

import gymnasium as gym
import numpy as np
import stable_baselines3
from stable_baselines3.common.callbacks import BaseCallback, tqdm
from stable_baselines3.common.vec_env import (DummyVecEnv, VecEnv, VecMonitor,
                                              is_vecenv_wrapped,
                                              sync_envs_normalization)


def evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    weights: np.ndarray = None,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            weights=weights,
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward


class EvalCallback(stable_baselines3.common.callbacks.EvalCallback):
    
    def __init__(self, eval_env: Union[gym.Env, VecEnv], callback_on_new_best: Optional[BaseCallback] = None, callback_after_eval: Optional[BaseCallback] = None, n_eval_episodes: int = 5, eval_freq: int = 10000, log_path: Optional[str] = None, best_model_save_path: Optional[str] = None, deterministic: bool = True, render: bool = False, verbose: int = 1, warn: bool = True, weights=None):
        super().__init__(eval_env, callback_on_new_best, callback_after_eval, n_eval_episodes, eval_freq, log_path, best_model_save_path, deterministic, render, verbose, warn)
        self.weights = weights

    def _init_callback(self) -> None:
        super()._init_callback()
        if self.weights is None:
            weights = np.zeros(self.model.scheduler.reward_dim)
            weights[-1] = 1
        else:
            weights = np.array(self.weights)
        self.weights = weights
        
    def _on_step(self) -> bool:
        continue_training = True
        
        start = time()

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                weights=self.weights,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        self.model.evaluation_time += time() - start

        return continue_training


class ProgressBarCallback(BaseCallback):
    """
    Display a progress bar when training SB3 agent
    using tqdm and rich packages.
    """

    def __init__(self) -> None:
        super().__init__()
        if tqdm is None:
            raise ImportError(
                "You must install tqdm and rich in order to use the progress bar callback. "
                "It is included if you install stable-baselines with the extra packages: "
                "`pip install stable-baselines3[extra]`"
            )
        self.pbar = None

    def _on_training_start(self) -> None:
        # Initialize progress bar
        # Remove timesteps that were done in previous training sessions
        self.pbar = tqdm(total=self.locals["total_timesteps"] - self.model.num_timesteps)
        self.prev_num_timesteps = self.model.num_timesteps

    def _on_step(self) -> bool:
        # Update progress bar, we do num_envs steps per call to `env.step()`
        if self.prev_num_timesteps != self.model.num_timesteps:
            self.pbar.update(self.model.num_timesteps - self.prev_num_timesteps)
            self.prev_num_timesteps = self.model.num_timesteps
        return True

    def _on_training_end(self) -> None:
        # Flush and close progress bar
        self.pbar.refresh()
        self.pbar.close()


class SchedulerCallback(BaseCallback):
    
    def __init__(self, config):
        super(SchedulerCallback, self).__init__()
        self.config = config
        
    def _init_callback(self):
        self.scheduler = self.model.scheduler
        self.scheduler.init_period()
        # self.was_agent_updated = False
        
        self.model._prev_object = None
        
    def _on_rollout_start(self) -> None:
        self.reward_sums = np.zeros(self.scheduler.reward_dim)
        self.rollout_episodes = 0
        
        current_weights = self.scheduler.get_current_weights()
        for reward_index in range(self.scheduler.reward_dim):
            reward_name = str(reward_index) if reward_index < self.scheduler.reward_dim - 1 else "main"
            self.logger.record(f"rollout/weight_{reward_name}", current_weights[reward_index])
            
        self.model._prev_object = None
        
    def _on_step(self):
        
        start = time()
        
        cl_add_idx_list = []
        was_curriculum_updated = False
        for idx, done in enumerate(self.locals["dones"]):
            
            if done:
                self.model._prev_object = None
                
                assert "episode" in self.locals["infos"][idx] and "r_separate" in self.locals["infos"][idx]["episode"], "Did not find individual rewards summed up in the `info` variable. Check if your environment is wrapped in the right Monitor."
                r_separate = self.locals["infos"][idx]["episode"]["r_separate"]
                
                # save the rewards for logging later
                self.reward_sums += r_separate
                self.rollout_episodes += 1
                
                # maybe update the curriculum learning weight distribution
                # new observation is automatically the initial one when episode terminates; this is how SB3 handles resets.
                was_curriculum_updated = self.scheduler.maybe_update(episode_rewards=r_separate,
                                                                     initial_observation=self.locals["new_obs"],
                                                                     learner=self.model,
                                                                     num_timesteps=self.model.num_timesteps,
                                                                     total_timesteps=self.model._total_timesteps)
                cl_add_idx_list.append(idx)
            
                if was_curriculum_updated:
                    break
        
        if was_curriculum_updated:
            # self.was_agent_updated = False
            self.model._last_obs = self.model.env.reset()
            
            assert len(cl_add_idx_list) > 0
            # discard all experience collected in other environments:
            self._safely_reset_buffers(set(range(len(self.locals["dones"]))) - set(cl_add_idx_list))
            
        self.model.cl_train_time += time() - start
                
        return True
            
    def _on_rollout_end(self) -> None:
        self.was_agent_updated = True
        self.model._last_obs = self.model.env.reset()

        for reward_index in range(self.scheduler.reward_dim):
            reward_name = str(reward_index) if reward_index < self.scheduler.reward_dim - 1 else "main"
            self.logger.record(f"rollout/reward_{reward_name}", self.reward_sums[reward_index] / self.rollout_episodes)
        
    def _safely_reset_buffers(self, idx_list):
        for idx in idx_list:
            if "dones" in self.locals:
                self.locals["dones"][idx] = False
            
                self.locals["action_mini_experience_buffer"][idx] = []
                self.locals["obs_mini_experience_buffer"][idx] = []
                self.locals["last_obs_mini_experience_buffer"][idx] = []
                self.locals["reward_mini_experience_buffer"][idx] = []
                self.locals["done_mini_experience_buffer"][idx] = []
                self.locals["info_mini_experience_buffer"][idx] = []
            
            if self.locals["action_noise"] is not None:
                kwargs = dict(indices=[idx]) if self.model.env.num_envs > 1 else {}
                self.locals["action_noise"].reset(**kwargs)
        