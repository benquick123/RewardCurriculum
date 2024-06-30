import numpy as np
from collections import OrderedDict

from cl_algorithms.scheduler import Scheduler


class ManualCurriculum(Scheduler):
    
    def __init__(self, reward_dim, tau=10, seed=None, update_frequency=20, sampling_strategy="random"):
        super().__init__(reward_dim, tau, seed, update_frequency)
        
        assert sampling_strategy in {"random", "current"}
        self.sampling_strategy = sampling_strategy
        
        self.level_times = [
            (0.0, 0.04), 
            (0.04, 0.095),
            (0.095, 0.4),
            (0.4, 0.8),
            (0.8, 0.9),
            (0.9, 0.91),
            (0.91, 0.95),
            (0.95, 0.96),
            (0.96, 1.0)
        ]
        
        self.level_weights = [
            (np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), ),
            ("fade", (0, 0), (0, 1)),
            (np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])),
            ("fade", (0, 0), (1, 1)),
            (np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])),
            ("fade", (0, 0), (1, 0)),
            (np.array([0.0, 0.0, 0.0, 0.5, 0.5, 0.0]), ),
            ("fade", (0, 0)),
            (np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), )
        ]
        
        self.current_progress = 0.0
        self.current_level_idx = 0
        self.current_sample = self.level_weights[self.current_level_idx]
    
    def maybe_update(self, **kwargs):
        assert "total_timesteps" in kwargs
        assert "num_timesteps" in kwargs
        
        self.current_progress = kwargs["num_timesteps"] / kwargs["total_timesteps"]
        
        self.update()
        self.init_period()
    
    def update(self):
        self.current_level_idx = 0
        while self.current_progress > self.level_times[self.current_level_idx][1]:
            self.current_level_idx += 1
            
        if isinstance(self.level_weights[self.current_level_idx][0], str) and self.level_weights[self.current_level_idx][0] == "fade":
            level_perc = (self.current_progress - self.level_times[self.current_level_idx][0]) / (self.level_times[self.current_level_idx][1] - self.level_times[self.current_level_idx][0])
            assert 0.0 <= level_perc <= 1.0, f"Error in `level_perc` computation: {level_perc}, {self.current_progress}, {self.level_times[self.current_level_idx][0]}, {self.level_times[self.current_level_idx][1]}"
            self.current_sample = []
            
            for prev_idx, next_idx in self.level_weights[self.current_level_idx][1:]:
                faded_weight = (1 - level_perc) * self.level_weights[self.current_level_idx - 1][prev_idx] + level_perc * self.level_weights[self.current_level_idx + 1][next_idx]
                self.current_sample.append(faded_weight)
                
            self.current_sample = np.array(self.current_sample)
        else:
            self.current_sample = self.level_weights[self.current_level_idx]
    
    def sample(self):
        self.current_weights = self.rng.choice(self.current_sample)
    
    def sample_batch(self, batch_size):
        if self.sampling_strategy == "random":
            return self.rng.random(size=(batch_size, self.reward_dim))
        else:
            return self.rng.choice(self.current_sample, size=batch_size, replace=True)
            