from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from torch.nn import functional as F
from gymnasium import spaces
from stable_baselines3.common.distributions import (
    SquashedDiagGaussianDistribution, StateDependentNoiseDistribution)
from stable_baselines3.common.policies import BasePolicy, BaseModel
from stable_baselines3.common.preprocessing import (get_action_dim,
                                                    get_flattened_obs_dim)
from stable_baselines3.common.torch_layers import (BaseFeaturesExtractor,
                                                   FlattenExtractor, CombinedExtractor,
                                                   create_mlp)
from stable_baselines3.common.type_aliases import Schedule
from torch import nn as nn

from rl_algorithms.tqc_policies import Actor, Critic, TQCPolicy, LOG_STD_MIN, LOG_STD_MAX


class SelectorFeatureExtractor(BaseFeaturesExtractor):
    
    def __init__(self, 
                 observation_space: spaces.Space,
                 reduced_observation_space: spaces.Space,
                 object_indices: List[Dict[str, np.ndarray]],
                 selector_net_arch: List[int],
                 baseline_features_extractor_class: nn.Module,
                 baseline_features_extractor_kwargs: Dict[str, Any] | None = None, 
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 selector_temperature: float = 1.0,
                 ) -> None:
        super().__init__(observation_space, get_flattened_obs_dim(reduced_observation_space))
        
        if baseline_features_extractor_kwargs is None:
            baseline_features_extractor_kwargs = {}
            
        self.baseline_features_extractor_class = baseline_features_extractor_class
        self.baseline_features_extractor_kwargs = baseline_features_extractor_kwargs
        self.object_indices = object_indices
        self.selector_net_arch = selector_net_arch
        self.activation_fn = activation_fn
        self.selector_temperature = selector_temperature
        self.num_objects = len(self.object_indices)
        self.input_features_dim = get_flattened_obs_dim(observation_space)
        
        self.baseline_features_extractor = baseline_features_extractor_class(observation_space, **baseline_features_extractor_kwargs)
        
        selector_net = create_mlp(self.input_features_dim, 
                                  self.num_objects, 
                                  self.selector_net_arch, 
                                  activation_fn)
        self.selector = nn.Sequential(*selector_net)
        
    def forward(self, obs: th.Tensor, deterministic: bool = False):
        flattened_obs = self.baseline_features_extractor(obs)
        selector_logits = self.selector(flattened_obs)
        
        # Apply Gumbel-Softmax trick to selector logits
        if deterministic:
            gumbel_noise = th.zeros_like(selector_logits)
        else:
            gumbel_noise = -th.log(-th.log(th.rand_like(selector_logits)))
        sampled = F.softmax((selector_logits + gumbel_noise) / self.selector_temperature, dim=-1)

        # Straight-Through Estimator: hard in forward, soft in backward
        hard_sampled = th.zeros_like(sampled).scatter_(-1, sampled.max(dim=-1, keepdim=True)[1], 1.0)
        hard_sampled = (hard_sampled - sampled).detach() + sampled
        detached_hard_sampled = hard_sampled.detach()
        
        # Process each part and aggregate
        sliced_obs = []
        for indices in self.object_indices:
            _obs = th.cat([obs[key][:, indices[key]] for key in obs.keys()] + [detached_hard_sampled], dim=-1).float()
            sliced_obs.append(_obs)
            
        sliced_obs = th.stack(sliced_obs, dim=1)
        features = (sliced_obs * hard_sampled.unsqueeze(-1)).sum(1)                
        return features


class SelectorMlpPolicy(TQCPolicy):
    
    def __init__(self, 
                 observation_space: spaces.Space, 
                 action_space: spaces.Box, 
                 lr_schedule: Schedule, 
                 net_arch: List[int] | Dict[str, List[int]] | None = None, 
                 activation_fn: Type[nn.Module] = nn.ReLU, 
                 use_sde: bool = False, 
                 log_std_init: float = -3, 
                 use_expln: bool = False, 
                 clip_mean: float = 2, 
                 _features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor, 
                 _features_extractor_kwargs: Dict[str, Any] | None = None, 
                 normalize_images: bool = True, 
                 optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam, 
                 optimizer_kwargs: Dict[str, Any] | None = None, 
                 n_quantiles: int = 25, 
                 n_critics: int = 2, 
                 use_retrospective_loss: bool = False, 
                 share_features_extractor: bool = False):
        
        share_features_extractor = True
        
        object_indices = [
            {"observation": np.concatenate((np.arange(0, 7), np.arange(7, 19))), 
             "achieved_goal": np.concatenate((np.arange(0, 3), np.arange(3, 6))), 
             "desired_goal": np.concatenate((np.arange(0, 3), np.arange(3, 6))), 
             "weights": np.arange(0, 6)},
            {"observation": np.concatenate((np.arange(0, 7), np.arange(19, 31))),
             "achieved_goal": np.concatenate((np.arange(0, 3), np.arange(6, 9))), 
             "desired_goal": np.concatenate((np.arange(0, 3), np.arange(6, 9))),
             "weights": np.arange(0, 6)}
        ]
        
        features_extractor_kwargs = {}
        features_extractor_kwargs["baseline_features_extractor_class"] = _features_extractor_class
        features_extractor_kwargs["baseline_features_extractor_kwargs"] = _features_extractor_kwargs
        features_extractor_kwargs["object_indices"] = object_indices
        features_extractor_kwargs["selector_net_arch"] = net_arch["pi"].copy()
        features_extractor_class = SelectorFeatureExtractor
        
        reduced_observation_space_entries = dict()
        for key in observation_space.keys():
            shape = object_indices[0][key].shape[0]
            # NOTE: this only works with observation spaces having the same value range for all observation dimensions.
            highs = observation_space[key].high[:shape]
            lows = observation_space[key].low[:shape]
            reduced_observation_space_entries[key] = spaces.Box(low=lows, high=highs, shape=(shape, ))
            
        reduced_observation_space_entries["object_idx"] = spaces.Box(low=0.0, high=1.0, shape=(len(object_indices), ))
        reduced_observation_space = spaces.Dict(reduced_observation_space_entries)
        features_extractor_kwargs["reduced_observation_space"] = reduced_observation_space
        
        super().__init__(observation_space, 
                         action_space, 
                         lr_schedule, 
                         net_arch, 
                         activation_fn, 
                         use_sde, 
                         log_std_init, 
                         use_expln, 
                         clip_mean, 
                         features_extractor_class, 
                         features_extractor_kwargs, 
                         normalize_images, 
                         optimizer_class, 
                         optimizer_kwargs, 
                         n_quantiles, 
                         n_critics, 
                         use_retrospective_loss,
                         share_features_extractor)
        
    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return Actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Critic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return Critic(**critic_kwargs).to(self.device)
    

class MultiInputSelectorPolicy(SelectorMlpPolicy):
    
    def __init__(self, 
                 observation_space: spaces.Space, 
                 action_space: spaces.Box, 
                 lr_schedule: Schedule, 
                 net_arch: List[int] | Dict[str, List[int]] | None = None, 
                 activation_fn: Type[nn.Module] = nn.ReLU, 
                 use_sde: bool = False, 
                 log_std_init: float = -3, 
                 use_expln: bool = False, 
                 clip_mean: float = 2, 
                 features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor, 
                 features_extractor_kwargs: Dict[str, Any] | None = None, 
                 normalize_images: bool = True, optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam, 
                 optimizer_kwargs: Dict[str, Any] | None = None, 
                 n_quantiles: int = 25, 
                 n_critics: int = 2, 
                 use_retrospective_loss: bool = False, 
                 share_features_extractor: bool = False):
        
        super().__init__(observation_space, 
                         action_space, 
                         lr_schedule, 
                         net_arch, 
                         activation_fn, 
                         use_sde, 
                         log_std_init, 
                         use_expln, 
                         clip_mean, 
                         features_extractor_class, 
                         features_extractor_kwargs, 
                         normalize_images, 
                         optimizer_class, 
                         optimizer_kwargs, 
                         n_quantiles, 
                         n_critics, 
                         use_retrospective_loss, 
                         share_features_extractor)