from datetime import datetime

import numpy as np
import random
from sklearn.mixture import GaussianMixture

from cl_algorithms.scheduler import Scheduler


class ALPGMM(Scheduler):
    
    def __init__(self, reward_dim, 
                 update_frequency, 
                 update_weights_frequency=5, 
                 tau=20, 
                 main_reward_objective_weight=5, 
                 max_history_size=None,
                 nb_neighbours=3, 
                 fitting_buffer_size=50, 
                 nb_bootstrap=5, 
                 seed=None, 
                 sampling_strategy="past", 
                 random_task_ratio=0.2, 
                 gmm_fitness_fn="aic", 
                 potential_clusters=[2, 11, 1], 
                 gmm_kwargs={}):
        
        super(ALPGMM, self).__init__(reward_dim=reward_dim, tau=tau, seed=seed, update_frequency=update_frequency)
        
        self.update_weights_frequency = update_weights_frequency
        self.fitting_buffer_size = fitting_buffer_size
        self.nb_bootstrap = max(nb_bootstrap, potential_clusters[1] - 1)
        self.random_task_ratio = random_task_ratio
        self.nb_neighbours = nb_neighbours
        
        self.main_reward_objective_weight = main_reward_objective_weight
        self.max_history_size = max_history_size
        
        assert sampling_strategy in {"past", "random"}
        self.sampling_strategy = sampling_strategy
        
        assert gmm_fitness_fn in {"aic", "bic"}
        self.gmm_fitness_fn = gmm_fitness_fn
        
        gmm_kwargs["random_state"] = self.seed
        self.gmms = [GaussianMixture(n_components=k, **gmm_kwargs) for k in np.arange(*potential_clusters)]
        self.gmm = None
        
        self.history = []
        self.distance_matrix = np.empty((0, 0))
        
        self.n_maybe_update = 0
        
    def init_period(self):
        self.period_rewards = []
        return super().init_period()
        
    def maybe_update(self, **kwargs):
        assert "episode_rewards" in kwargs
        # assert "was_agent_updated" in kwargs
        
        self.period_rewards.append(kwargs["episode_rewards"])
        
        if (self.n_maybe_update + 1) % self.update_weights_frequency == 0:
            self.update_history()
            # resample the weights
            self.init_period()
        
        if (self.n_maybe_update + 1) % self.update_frequency == 0:
            # update the weight distribution
            self.update()
            was_curriculum_updated = True
        else:
            was_curriculum_updated = False
            
        self.n_maybe_update += 1

        return was_curriculum_updated
    
    def update_history(self):
        # the new distances are added to the distance matrix.
        # this has to be done before the weight update.
        distances = []
        for weights, _, _ in self.history:
            distances.append(np.sqrt(np.sum((weights - self.get_current_weights())**2)))
        distances.append(0)
        distances = np.array(distances).reshape(1, -1)
        
        # extend the distance matrix
        assert len(distances[0]) == len(self.distance_matrix) + 1
        self.distance_matrix = np.concatenate([self.distance_matrix, distances[:, :-1]], axis=0)
        self.distance_matrix = np.concatenate([self.distance_matrix, distances.T], axis=1)
        
        self.period_rewards = np.stack(self.period_rewards, axis=0)
        self.period_rewards = self.period_rewards.mean(axis=0)
        
        # when enough neighbours in the history, compute the objective and save new entry
        if len(self.distance_matrix) >= self.nb_neighbours + 1:
            
            weighted_rewards = (self.period_rewards @ self.get_current_weights().reshape(-1, 1))[0]
            
            # argsort the distances to obtain closest neighbours
            # obtain weighted rewards for these neighbours
            distance_indices = np.argsort(distances[0, :-1])[:self.nb_neighbours].astype(np.int32)
            # compute previous close weightted rewards
            previous_weighted_rewards = np.mean([self.history[idx][1] @ self.history[idx][0].reshape(-1, 1) for idx in distance_indices])
            
            alp = np.abs(weighted_rewards - previous_weighted_rewards)
            
            max_distance = np.linalg.norm(np.ones(self.reward_dim))
            main_rewards_vector = np.array([0 for _ in range(self.reward_dim - 1)] + [1])
            main_reward_vector_distance = np.linalg.norm((main_rewards_vector - self.get_current_weights()))
            
            main_reward = self.period_rewards[-1]
            objective = alp + self.main_reward_objective_weight * (1 - main_reward_vector_distance / max_distance) * main_reward
        else:
            objective = 0
        
        history_entry = (self.get_current_weights(), self.period_rewards, objective)
        self.history.append(history_entry)
        if self.max_history_size is not None and len(self.history) > self.max_history_size:
            self.history = self.history[-self.max_history_size:]
    
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
            # self.current_weights = np.exp(self.current_weights * self.tau) / np.exp(self.current_weights * self.tau).sum()
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
                    idx = np.argmax(probas)
                    # idx = np.where(self.rng.multinomial(1, probas) == 1)[0][0]
                except ValueError:
                    print(probas)
                
            # 3 - Sample task in Gaussian, without forgetting to remove objective dimension
            self.current_weights = self.rng.multivariate_normal(self.gmm.means_[idx], self.gmm.covariances_[idx])[:-1]
            self.current_weights = np.clip(self.current_weights, 0.0, 1.0)
            
        # print("----------------- WEIGHTS CHANGED --------------------")
        
    def sample_batch(self, batch_size):
        if self.sampling_strategy == "past":
            reward_weights = np.array([w for w, _, _ in random.choices(self.history, k=batch_size)])
        else:
            reward_weights = self.rng.random((batch_size, self.reward_dim))
            # reward_weights = np.exp(reward_weights * self.tau) / np.exp(reward_weights * self.tau).sum(axis=1, keepdims=True)
        return reward_weights