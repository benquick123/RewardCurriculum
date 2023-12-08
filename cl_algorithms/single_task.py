import numpy as np

from cl_algorithms.scheduler import Scheduler


class SingleTask(Scheduler):
    
    def __init__(self, reward_dim, seed=None, tau=20, update_frequency=20, **kwargs):
        super(SingleTask, self).__init__(reward_dim=reward_dim, tau=tau, seed=seed, update_frequency=update_frequency)
        
    def sample(self):
        if self.current_weights is None:
            self.current_weights = np.zeros(self.reward_dim)
            self.current_weights[-1] = 1
        
    def sample_batch(self, batch_size):
        weight_batch = np.array(self.current_weights).reshape(1, -1)
        weight_batch = np.repeat(weight_batch, batch_size, axis=0)
        return weight_batch
    
    def maybe_update(self, **kwargs):
        return False


