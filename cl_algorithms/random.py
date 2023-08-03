import os

import numpy as np


class Random:
    
    def __init__(self, reward_dim, learner, tau=20, seed=None):
        self.reward_dim = reward_dim
        self.tau = tau
        # self.learner = learner
        
        self.seed = seed if seed is not None else np.random.randint(0, 9999999)
        self.rng = np.random.default_rng(self.seed)
        
        self.all_weights = dict()
        self.all_weights_array = list()
        
        self.sample()
        
    def init_period(self, *args, **kwargs):
        return
    
    def end_period(self, *args, **kwargs):
        self.all_weights[hash(np.round(self.current_weights, 2).tobytes())] = self.current_weights
    
    def sample(self):
        self.current_weights = self.rng.random(self.reward_dim)
        self.current_weights = np.exp(self.current_weights * self.tau) / np.exp(self.current_weights * self.tau).sum()
        
    def sample_multiple(self, n_samples):
        if len(self.all_weights) != len(self.all_weights_array):
            self.all_weights_array = np.stack(list(self.all_weights.values()), axis=0)
            
        indices = self.rng.choice(len(self.all_weights_array), size=n_samples)
        weight_batch = self.all_weights_array[indices]
        
        # weight_batch = self.rng.uniform(0, 1, size=(n_samples, self.reward_dim))
        # weight_batch = np.exp(weight_batch * self.tau) / np.exp(weight_batch * self.tau).sum(axis=1, keepdims=True)
        return weight_batch
    
    def save_reward(self, *args, **kwargs):
        return

    def log(self, log_path, additional_data=[], logger=None):
        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write(",".join(["w" + str(i) for i in range(self.reward_dim)]) + "\n")
                
        with open(log_path, "a") as f:
            f.write(",".join(list(map(str, self.current_weights))))
            if additional_data:
                f.write("," + ",".join(list(map(str, additional_data))))
            f.write("\n")
            
    def set_attrs(self, *args, **kwargs):
        return