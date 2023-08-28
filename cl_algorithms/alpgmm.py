from datetime import datetime

import numpy as np
from sklearn.mixture import GaussianMixture

from cl_algorithms.scheduler import Scheduler


class ALPGMM(Scheduler):
    
    def __init__(self, reward_dim, update_frequency, tau=20, nb_neighbours=3, fitting_buffer_size=50, nb_bootstrap=5, seed=None, random_task_ratio=0.2, gmm_fitness_fn="aic", potential_clusters=[2, 11, 1], gmm_kwargs={}):
        super(ALPGMM, self).__init__(reward_dim=reward_dim, tau=tau, seed=seed, update_frequency=update_frequency)
        
        self.fitting_buffer_size = fitting_buffer_size
        self.nb_bootstrap = max(nb_bootstrap, potential_clusters[1] - 1)
        self.random_task_ratio = random_task_ratio
        self.nb_neighbours = nb_neighbours
        
        assert gmm_fitness_fn in {"aic", "bic"}
        self.gmm_fitness_fn = gmm_fitness_fn
        
        gmm_kwargs["random_state"] = self.seed
        self.gmms = [GaussianMixture(n_components=k, **gmm_kwargs) for k in np.arange(*potential_clusters)]
        self.gmm = None
        
        self.history = []
        self.distance_matrix = np.empty((0, 0))
        
    def maybe_update(self, **kwargs):
        assert "episode_rewards" in kwargs
        # assert "was_agent_updated" in kwargs
        
        distances = []
        for weights, _, _ in self.history:
            distances.append(np.sqrt(np.sum((weights - self.current_weights)**2)))
        distances.append(0)
        distances = np.array(distances).reshape(1, -1)
        
        # extend the distance matrix
        assert len(distances[0]) == len(self.distance_matrix) + 1
        self.distance_matrix = np.concatenate([self.distance_matrix, distances[:, :-1]], axis=0)
        self.distance_matrix = np.concatenate([self.distance_matrix, distances.T], axis=1)
        
        if len(distances[0]) >= self.nb_neighbours + 1:
            weighted_rewards = (kwargs["episode_rewards"] @ self.current_weights.reshape(-1, 1))[0]
            
            # argsort the distances to obtain closest neighbours
            # obtain weighted rewards for these neighbours
            distance_indices = np.argsort(distances[0, :-1])[:self.nb_neighbours].astype(np.int32)
            # compute previous close weightted rewards
            previous_weighted_rewards = np.mean([self.history[idx][1] @ self.history[idx][0].reshape(-1, 1) for idx in distance_indices])
            
            alp = np.abs(weighted_rewards - previous_weighted_rewards)
            main_reward = kwargs["episode_rewards"][-1]
            objective = alp + main_reward
        else:
            objective = 0
        
        history_entry = (self.current_weights, kwargs["episode_rewards"], objective)
        self.history.append(history_entry)
        
        # print("3. UPDATE CL:", self.get_current_weights())
        
        if len(self.history) % self.update_frequency == 0:
            self.update()
            was_agent_updated = True
        else:
            was_agent_updated = False

        self.init_period()
        return was_agent_updated
        
    def init_period(self):
        self.sample()
    
    def update(self): 
        if len(self.history) > self.nb_bootstrap:
            train_set = np.stack([np.concatenate([weights, [objective]]) for weights, _, objective in self.history[-self.fitting_buffer_size:]], axis=0)
            self.gmms = [g.fit(train_set) for g in self.gmms]
            fitnesses = [getattr(g, self.gmm_fitness_fn)(train_set) for g in self.gmms]
            self.gmm = self.gmms[np.argmin(fitnesses)]
    
    def sample(self):        
        # sample new weights
        if len(self.history) < self.nb_bootstrap or self.rng.random() < self.random_task_ratio or self.gmm is None:
            self.current_weights = self.rng.random(self.reward_dim)
            # normalize the weights
            self.current_weights = np.exp(self.current_weights * self.tau) / np.exp(self.current_weights * self.tau).sum()
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
            self.current_weights = np.clip(self.current_weights, 0.0, 1.0)
            
        # print("----------------- WEIGHTS CHANGED --------------------")
        
        
    def sample_batch(self, batch_size, from_history_only=True):
        reward_weights = self.rng.random((batch_size, self.reward_dim))
        reward_weights = np.exp(reward_weights * self.tau) / np.exp(reward_weights * self.tau).sum(axis=1, keepdims=True)
        return reward_weights