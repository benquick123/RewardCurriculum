import numpy as np

from cl_algorithms.scheduler import Scheduler


class SACX(Scheduler):
    
    def __init__(self, reward_dim, tau=10, seed=None, update_frequency=20, reward_history_size=100, sampling_strategy="random"):
        # notice, update_frequency is in steps now.
        super().__init__(reward_dim, tau, seed, update_frequency)
        
        self.reward_history = [None] * reward_dim
        self.reward_history_size = reward_history_size
        
        self.distribution = None
        
        assert sampling_strategy in {"random", "current"}
        self.sampling_strategy = sampling_strategy
    
    def maybe_update(self, **kwargs):
        assert "episode_rewards" in kwargs
        
        main_reward = kwargs["episode_rewards"][-1]
        
        weight_idx = np.argmax(self.get_current_weights())
        if self.reward_history[weight_idx] is None:
            self.reward_history[weight_idx] = list()
        
        self.reward_history[weight_idx].append(main_reward)
        if len(self.reward_history[weight_idx]) > self.reward_history_size:
            self.reward_history[weight_idx] = self.reward_history[weight_idx][1:]
            
        self.update()
        self.init_period()
        
    def update(self):
        self.distribution = np.zeros(self.reward_dim)
        for idx, rewards in enumerate(self.reward_history):
            if rewards is not None:
                self.distribution[idx] = np.mean(rewards)
            
        self.distribution += 1e-8
        self.distribution *= self.tau
        self.distribution = np.exp(self.distribution)
        self.distribution /= self.distribution.sum()
    
    def sample(self):
        self.current_weights = np.zeros(self.reward_dim)
        weight_idx = self.rng.choice(np.arange(self.reward_dim), p=self.distribution)
        self.current_weights[weight_idx] = 1.0
    
    def sample_batch(self, batch_size):
        weight_batch = np.zeros((batch_size, self.reward_dim))
        if self.sampling_strategy == "current":
            idxs = self.rng.choice(np.arange(self.reward_dim), size=batch_size, p=self.distribution)
        else:
            idxs = self.rng.integers(0, self.reward_dim, size=batch_size)
        weight_batch[np.arange(batch_size), idxs] = 1.0
        
        return weight_batch
    
    def get_current_weights(self):
        return self.current_weights.copy()