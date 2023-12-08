import numpy as np

from cl_algorithms.scheduler import Scheduler


class ManualTask(Scheduler):
    
    def __init__(self, reward_dim, reward_weights=None, seed=None, tau=20, update_frequency=20, **kwargs):
        assert reward_weights is not None, "reward_weights should not be None."
        assert np.sum(reward_weights) > 0, "At least one weight has to be larger than 0"
        assert len(reward_weights) == reward_dim
        
        self.reward_weights = np.array(reward_weights)
        assert np.all(self.reward_weights >= 0)
        
        self.reward_weights /= np.sum(self.reward_weights)
        super(ManualTask, self).__init__(reward_dim=reward_dim, tau=tau, seed=seed, update_frequency=update_frequency)
        
    def sample(self):
        if self.current_weights is None:
            self.current_weights = np.array(self.reward_weights)
        
    def sample_batch(self, batch_size):
        # weight_batch = np.array(self.current_weights).reshape(1, -1)
        # weight_batch = np.repeat(weight_batch, batch_size, axis=0)
        # return weight_batch
        weights = self.rng.random((batch_size, self.reward_dim))
        return np.exp(weights * self.tau) / np.exp(weights * self.tau).sum(axis=1, keepdims=True)
    
    def maybe_update(self, **kwargs):
        return False
