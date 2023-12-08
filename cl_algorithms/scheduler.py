import numpy as np

class Scheduler:
    
    def __init__(self, reward_dim, tau=10, seed=None, update_frequency=20):
        self.reward_dim = reward_dim
        self.tau = tau
        self.update_frequency = update_frequency
        self.seed = seed if seed is not None else np.random.randint(0, 9999999)
        self.rng = np.random.default_rng(self.seed)
        self.current_weights = None
    
    def maybe_update(self, **kwargs):
        raise NotImplementedError
    
    def init_period(self):
        self.sample()
    
    def end_period(self):
        raise NotImplementedError
    
    def sample(self):
        raise NotImplementedError
    
    def sample_batch(self, batch_size):
        raise NotImplementedError
    
    def get_current_weights(self):
        return self.current_weights.copy()