import numpy as np

from cl_algorithms.scheduler import Scheduler


class Random(Scheduler):
    
    def __init__(self, reward_dim, tau=20, seed=None, sampling_strategy="past", **kwargs):
        assert sampling_strategy in {"past", "random"}
        self.sampling_strategy = sampling_strategy
        if self.sampling_strategy == "past":
            self.past_weights = list()
            
        super(Random, self).__init__(reward_dim=reward_dim, tau=tau, seed=seed, update_frequency=1)
        
    def maybe_update(self, *args, **kwargs):
        self.past_weights.append(self.current_weights)
        
        self.init_period()
        return False
    
    def sample(self):
        self.current_weights = self.rng.random(self.reward_dim)
        # normalize the weights
        # self.current_weights = np.exp(self.current_weights * self.tau) / np.exp(self.current_weights * self.tau).sum()
    
    def sample_batch(self, batch_size):
        if self.sampling_strategy == "past" and len(self.past_weights) > 0:
            return self.rng.choice(self.past_weights, size=batch_size)
        else:
            weights = self.rng.random((batch_size, self.reward_dim))
            # return np.exp(weights * self.tau) / np.exp(weights * self.tau).sum(axis=1, keepdims=True)
            return weights # np.exp(weights * self.tau) / np.exp(weights * self.tau).sum(axis=1, keepdims=True)
        