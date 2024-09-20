from copy import deepcopy
from functools import partial
from typing import ClassVar, Dict, Optional, Tuple, Type, Union, List, Any

from time import time

import numpy as np
import sb3_contrib
import stable_baselines3
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecEnv
import torch as th
from gymnasium import spaces
from sb3_contrib.common.utils import quantile_huber_loss
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import MaybeCallback, RolloutReturn, TrainFreq
from stable_baselines3.common.utils import polyak_update

from cl_algorithms.single_task import SingleTask
from rl_algorithms.replay_buffer import HerReplayBuffer, ReplayBuffer, DictReplayBuffer
from rl_algorithms.tqc_policies import (CnnPolicy, MlpPolicy, MultiInputPolicy,
                                        TQCPolicy)
from rl_algorithms.tqc_selector_policies import SelectorMlpPolicy, MultiInputSelectorPolicy
from utils.callbacks import ProgressBarCallback
from utils.retrospective_loss import retrospective_loss_fn
from utils.train_utils import compute_normalized_entropy


class TQC(sb3_contrib.TQC):
    
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "SelectorMlpPolicy": SelectorMlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
        "MultiInputSelectorPolicy": MultiInputSelectorPolicy,
    }
    policy: TQCPolicy
    
    def __init__(self, policy, env, use_uvfa=True, reward_dim=1, scheduler_class=SingleTask, scheduler_kwargs={}, use_retrospective_loss=False, **kwargs):
        self.use_uvfa = use_uvfa
        self.scheduler = scheduler_class(reward_dim=reward_dim, **scheduler_kwargs)
        
        kwargs["replay_buffer_class"] = kwargs.get("replay_buffer_class", "HerReplayBuffer")
        if isinstance(kwargs["replay_buffer_class"], str):
            if kwargs["replay_buffer_class"] == "HerReplayBuffer":
                kwargs["replay_buffer_class"] = HerReplayBuffer
            elif kwargs["replay_buffer_class"] == "ReplayBuffer":
                kwargs["replay_buffer_class"] = ReplayBuffer
            elif kwargs["replay_buffer_class"] == "DictReplayBuffer":
                kwargs["replay_buffer_class"] = DictReplayBuffer
            else:
                raise NotImplementedError(f"Replay buffer class {kwargs['replay_buffer_class']} not implemented.")
                
        if "replay_buffer_kwargs" not in kwargs:
            kwargs["replay_buffer_kwargs"] = dict()
        # kwargs["replay_buffer_kwargs"]["reward_dim"] = reward_dim
        kwargs["replay_buffer_kwargs"]["scheduler"] = self.scheduler
        kwargs["replay_buffer_kwargs"]["use_uvfa"] = use_uvfa
        
        self.use_retrospective_loss = use_retrospective_loss
        
        self._prev_object = None
        
        self.evaluation_time = 0
        self.simulation_time = 0
        self.rl_train_time = 0
        self.cl_train_time = 0
        
        super().__init__(policy, env, **kwargs)
        
    def _setup_model(self) -> None:
        super()._setup_model()
        
        if self.use_retrospective_loss:
            self.critic_retro = self.policy.critic_retro
            
    def collect_rollouts(self, env: VecEnv, callback: BaseCallback, train_freq: TrainFreq, replay_buffer: ReplayBuffer, action_noise: ActionNoise | None = None, learning_starts: int = 0, log_interval: int | None = None) -> RolloutReturn:
        start = time()
        ret = super().collect_rollouts(env, callback, train_freq, replay_buffer, action_noise, learning_starts, log_interval)
        self.simulation_time += time() - start
        return ret
            
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        
        start = time()
        
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        if hasattr(self.actor, "features_extractor_optimizer"):
            optimizers += [self.actor.features_extractor_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        selected_ents = []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            
            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            
            onehot_selected_indices = reduced_observations = onehot_selected_probabilities = None
            if hasattr(self.actor.features_extractor, "get_features") and hasattr(self.actor.features_extractor, "get_selected"):
                with th.no_grad():
                    # at no point is it necessary to run gradient through reduced observations, so we can sefely detach.
                    reduced_observations = self.actor.features_extractor.get_features().detach()
                    onehot_selected_indices = self.actor.features_extractor.get_selected(return_probs=False).detach()
                    onehot_selected_probabilities = self.actor.features_extractor.get_selected(return_probs=True).detach().cpu()
                    if not th.all(th.isfinite(onehot_selected_probabilities)):
                        print(onehot_selected_probabilities)
                        print(th.sum(~th.isfinite(onehot_selected_probabilities)))
                        exit()
                    selected_ents.append(np.mean(compute_normalized_entropy(onehot_selected_probabilities).numpy()))
            
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                
                # Compute and cut quantiles at the next state
                # batch x nets x quantiles
                if hasattr(self.actor.features_extractor, "features_from_indices") and \
                   hasattr(self.actor.features_extractor, "get_features") and \
                   hasattr(self.actor.features_extractor, "get_selected") and \
                   hasattr(self.actor, "action_log_prob_from_reduced_obs") and \
                   hasattr(self.critic_target, "value_from_reduced_obs"):
                       # take care of the potential selected object change in the future.
                       next_reduced_observations_same = self.actor.features_extractor.features_from_indices(replay_data.next_observations, onehot_selected_indices)
                       next_actions_same, next_log_prob_same = self.actor.action_log_prob_from_reduced_obs(next_reduced_observations_same)
                       
                       next_actions_diff, next_log_prob_diff = self.actor.action_log_prob(replay_data.next_observations)
                       next_reduced_observations_diff = self.actor.features_extractor.get_features()
                       
                       # get next probabilities and entropy
                       # NOTE: entropy is normalized and is always on interval [0, 1]
                       next_entropy = compute_normalized_entropy(self.actor.features_extractor.get_selected(return_probs=True))
                       
                       # change or keep the object the same; use randomness and entropy as a measure.
                       condition = (th.rand_like(next_entropy) > next_entropy).unsqueeze(-1)
                       next_actions = th.where(condition, next_actions_same, next_actions_diff)
                       next_log_prob = th.where(condition, next_log_prob_same.unsqueeze(-1), next_log_prob_diff.unsqueeze(-1))
                       next_reduced_observations = th.where(condition, next_reduced_observations_same, next_reduced_observations_diff)
                       
                       next_quantiles = self.critic_target.value_from_reduced_obs(next_reduced_observations, next_actions)
                else:
                    next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                    next_quantiles = self.critic_target(replay_data.next_observations, next_actions)

                # Sort and drop top k quantiles to control overestimation.
                n_target_quantiles = self.critic.quantiles_total - self.top_quantiles_to_drop_per_net * self.critic.n_critics
                next_quantiles, _ = th.sort(next_quantiles.reshape(batch_size, -1))
                next_quantiles = next_quantiles[:, :n_target_quantiles]

                # td error + entropy term
                target_quantiles = next_quantiles - ent_coef * next_log_prob.reshape(-1, 1)
                target_quantiles = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_quantiles
                # Make target_quantiles broadcastable to (batch_size, n_critics, n_target_quantiles).
                target_quantiles.unsqueeze_(dim=1)

            # Get current Quantile estimates using action from the replay buffer
            if hasattr(self.critic_target, "value_from_reduced_obs"):
                current_quantiles = self.critic.value_from_reduced_obs(reduced_observations, replay_data.actions.to(th.float32))
            else:
                current_quantiles = self.critic(replay_data.observations, replay_data.actions.to(th.float32))
            # Compute critic loss, not summing over the quantile dimension as in the paper.
            critic_loss = quantile_huber_loss(current_quantiles, target_quantiles, sum_over_quantiles=False)
            critic_losses.append(critic_loss.item())
            
            if self._n_updates > 0 and gradient_step > 0 and self.use_retrospective_loss:
                if self._n_updates == 1:
                    print("Retrospective loss has not been tested with TQC yet! Probably doesn't work.")
                    
                with th.no_grad():
                    retro_quantiles = self.critic_retro(replay_data.observations, replay_data.actions.to(th.float32))
                    
                retrospective_loss = sum([
                    retrospective_loss_fn(
                        current_quantiles[:, i, :].unsqueeze(1),
                        retro_quantiles[:, i, :].unsqueeze(1),
                        target_quantiles,
                        K=2, scaled=True, distance=lambda: partial(quantile_huber_loss, sum_over_quantiles=False)
                    ) for i in range(self.critic.n_critics)
                ])

                
                critic_loss += retrospective_loss

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            if hasattr(self.critic, "value_from_reduced_obs"):
                qf_pi = self.critic.value_from_reduced_obs(reduced_observations, actions_pi).mean(dim=2).mean(dim=1, keepdim=True)
            else:
                qf_pi = self.critic(replay_data.observations, actions_pi).mean(dim=2).mean(dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            
            # deal with the features extractor training
            if hasattr(self.actor, "features_extractor_optimizer") and \
               hasattr(self.actor.features_extractor, "get_selected") and \
               hasattr(self.actor.features_extractor, "num_objects") and \
               hasattr(self.actor.features_extractor, "features_from_indices") and \
               hasattr(self.critic, "value_from_reduced_obs"):
                   # obtain probabilities and log probabilities for current batch
                   _ = self.actor(replay_data.observations, deterministic=True)
                   selector_probabilities = self.actor.features_extractor.get_selected(return_probs=True)
                   selector_log_probabilities = selector_probabilities.clip(1e-8, 1).log()
                   
                   # get q-values for selecting each object for every sample in the batch
                   # # get_values for each object
                   qf_pi = []
                   for idx in range(self.actor.features_extractor.num_objects):
                       indices = th.zeros(batch_size, self.actor.features_extractor.num_objects).to(self.actor.device)
                       indices[:, idx] = 1
                       reduced_observations = self.actor.features_extractor.features_from_indices(replay_data.observations, indices)
                       actions_pi = self.actor(reduced_observations, deterministic=True)
                       qf_pi.append(self.critic.value_from_reduced_obs(reduced_observations, actions_pi).mean(dim=2).mean(dim=1, keepdim=True))
                       
                   qf_pi = th.cat(qf_pi, dim=1)
                
                   # compute selector loss
                   # using static alpha for now
                   # TODO: add option for setting alpha via config
                   selector_loss = (selector_probabilities * (0.1 * selector_log_probabilities - qf_pi)).sum(dim=1).mean()
                
                   # backprop
                   self.actor.features_extractor_optimizer.zero_grad()
                   selector_loss.backward()
                   self.actor.features_extractor_optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see https://github.com/DLR-RM/stable-baselines3/issues/996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
                
        if self.use_retrospective_loss:
            self.critic_retro.load_state_dict(self.critic.state_dict())

        self._n_updates += gradient_steps
        
        self.rl_train_time += time() - start

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        if len(selected_ents) > 0:
            self.logger.record("train/selected_ent", np.mean(selected_ents))
    
    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        obs = deepcopy(self._last_original_obs)
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])
                        
        obs["weights"] = self.scheduler.get_current_weights()

        replay_buffer.add(
            obs,
            next_obs,
            buffer_action,
            reward_,
            dones,
            infos,
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_
    
    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            unscaled_action, _ = self.predict(deepcopy(self._last_obs), weights=self.scheduler.get_current_weights(), deterministic=False)
            
            if self._prev_object is not None and hasattr(self.actor.features_extractor, "get_selected"):
                entropy = compute_normalized_entropy(self.actor.features_extractor.get_selected(return_probs=True))
                assert entropy.shape == (1, ), f"Wrong entropy shape. It has to be one-dimensional, but got shape {entropy.shape}."
                
                if th.rand_like(entropy) < entropy:
                    # we keep the object the same as in previous step
                    tensor_obs = deepcopy(self._last_obs)
                    if isinstance(tensor_obs, spaces.Box):
                        tensor_obs = np.concatenate((tensor_obs, self.scheduler.get_current_weights()), axis=-1)
                    else:
                        tensor_obs["weights"] = self.scheduler.get_current_weights()
                    tensor_obs, vectorized_env = self.policy.obs_to_tensor(tensor_obs)
                    
                    with th.no_grad():
                        reduced_obs = self.actor.features_extractor.features_from_indices(tensor_obs, self._prev_object)
                        unscaled_action = self.policy.actor(reduced_obs, deterministic=False)
                        
                    unscaled_action = unscaled_action.cpu().numpy().reshape((-1, *self.action_space.shape))

                    if isinstance(self.action_space, spaces.Box):
                        if self.policy.squash_output:
                            unscaled_action = self.policy.unscale_action(unscaled_action)
                        else:
                            unscaled_action = np.clip(unscaled_action, self.action_space.low, self.action_space.high)

                    # Remove batch dimension if needed
                    if not vectorized_env:
                        unscaled_action = unscaled_action.squeeze(axis=0)
                                    
            if hasattr(self.actor.features_extractor, "get_selected"):
                self._prev_object = self.actor.features_extractor.get_selected(return_probs=False)            

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action
            
    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False
    ) -> Tuple[int, BaseCallback]:
        
        total_timesteps, callback = super()._setup_learn(total_timesteps=total_timesteps, 
                             callback=callback, 
                             reset_num_timesteps=reset_num_timesteps, 
                             tb_log_name=tb_log_name, 
                             progress_bar=progress_bar)
        
        if progress_bar:
            for idx, _callback in enumerate(callback.callbacks):
                if isinstance(_callback, stable_baselines3.common.callbacks.ProgressBarCallback):
                    callback.callbacks[idx] = ProgressBarCallback()
        callback.init_callback(self)
            
        return total_timesteps, callback
    
    def predict(self, observation: np.ndarray, weights: np.ndarray = None, **kwargs) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        if self.use_uvfa:
            weights = weights.reshape((-1, self.scheduler.reward_dim))
            if len(weights) != len(observation):
                batch_size = observation.shape[0] if isinstance(observation, spaces.Box) else observation["observation"].shape[0]
                weights = np.repeat(weights, batch_size, axis=0)
                
            if isinstance(observation, spaces.Box):
                observation = np.concatenate((observation, weights), axis=-1)
            else:
                observation["weights"] = weights
            
        return self.policy.predict(observation, **kwargs)