from datetime import datetime

import numpy as np
from sklearn.mixture import GaussianMixture

from cl_algorithms.scheduler import Scheduler


class ALPGMM(Scheduler):
    
    def __init__(self, reward_dim, update_frequency, tau=20, history_len=50, nb_bootstrap=5, seed=None, random_task_ratio=0.2, gmm_fitness_fn="aic", potential_clusters=[2, 11, 1], gmm_kwargs={}):
        super(ALPGMM, self).__init__(reward_dim=reward_dim, tau=tau, seed=seed, update_frequency=update_frequency)
        
        self.history_len = history_len
        self.nb_bootstrap = max(nb_bootstrap, potential_clusters[1] - 1)
        self.random_task_ratio = random_task_ratio
        
        assert gmm_fitness_fn in {"aic", "bic"}
        self.gmm_fitness_fn = gmm_fitness_fn
        
        gmm_kwargs["random_state"] = self.seed
        self.gmms = [GaussianMixture(n_components=k, **gmm_kwargs) for k in np.arange(*potential_clusters)]
        self.gmm = None
        
        self.history = []
        
    def maybe_update(self, **kwargs):
        assert "episode_rewards" in kwargs
        assert "was_agent_updated" in kwargs
        
        if kwargs["was_agent_updated"] and len(self.r_after_update) < self.update_frequency // 2:
            self.r_after_update.append(kwargs["episode_rewards"])
        elif not kwargs["was_agent_updated"] and len(self.r_before_update) < self.update_frequency // 2:
            self.r_before_update.append(kwargs["episode_rewards"])
        
        if len(self.r_before_update) + len(self.r_after_update) >= (self.update_frequency // 2) * 2:
            self.end_period()
            self.init_period()
            return True
        
        return False
    
    def init_period(self):
        self.r_before_update = []
        self.r_after_update = []
        self.sample()
    
    def end_period(self):
        self.r_before_update = np.stack(self.r_before_update, axis=0)
        self.r_after_update = np.stack(self.r_after_update, axis=0)
        
        weighted_r_before = self.r_before_update @ self.current_weights.reshape(-1, 1)
        weighted_r_after = self.r_after_update @ self.current_weights.reshape(-1, 1)
        
        alp = np.abs(weighted_r_before.mean() - weighted_r_after.mean())
        main_reward = np.concatenate([self.r_before_update[:, -1], self.r_after_update[:, -1]]).mean()
        objective = alp + main_reward
        
        history_entry = np.concatenate((self.current_weights, np.array([objective])))
        self.history.append(history_entry)
            
        if len(self.history) > self.nb_bootstrap:
            train_set = np.stack(self.history[-self.history_len:], axis=0)
            self.gmms = [g.fit(train_set) for g in self.gmms]
            fitnesses = [getattr(g, self.gmm_fitness_fn)(train_set) for g in self.gmms]
            self.gmm = self.gmms[np.argmin(fitnesses)]
    
    def sample(self):
        if len(self.history) < self.nb_bootstrap or self.rng.random() < self.random_task_ratio or self.gmm is None:
            self.current_weights = self.rng.random(self.reward_dim)
        else:
            # Objective-based task sampling
            # 1 - Retrieve the mean objective value of each Gaussian in the GMM
            objective_means = [np.abs(pos[-1]) for pos in self.gmm.means_]

            # 2 - Sample Gaussian proportionally to its mean objective
            if np.sum(objective_means) == 0:
                idx = self.rng.integers(0, len(objective_means))
            else:
                probas = np.array(objective_means) / (np.sum(objective_means) + 1e-10)
                try:
                    idx = np.where(self.rng.multinomial(1, probas) == 1)[0][0]
                except ValueError:
                    print(probas)
                
            # 3 - Sample task in Gaussian, without forgetting to remove objective dimension
            self.current_weights = self.rng.multivariate_normal(self.gmm.means_[idx], self.gmm.covariances_[idx])[:-1]
        
        # normalize the weights
        self.current_weights = np.exp(self.current_weights * self.tau) / np.exp(self.current_weights * self.tau).sum()
        
    def sample_batch(self, batch_size, from_history_only=True):
        reward_weights = self.rng.random((batch_size, self.reward_dim))
        reward_weights = np.exp(reward_weights * self.tau) / np.exp(reward_weights * self.tau).sum(axis=1, keepdims=True)
        return reward_weights