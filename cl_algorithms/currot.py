import os
import torch as th
import numpy as np
from torch.nn import functional as F
import pickle
from typing import Tuple, List
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist
import nadaraya_watson as na
import torchrl

from cl_algorithms.scheduler import Scheduler


def sliced_wasserstein(source, target, n=100, grad=False):
    # Generate random projection vectors
    dim = source.shape[1]
    directions = np.random.normal(0, 1, (n, dim))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

    # Compute the projected assignments
    source_projections = np.einsum("nd,md->nm", directions, source)
    target_projections = np.einsum("nd,md->nm", directions, target)

    sorted_source = np.argsort(source_projections, axis=-1)
    reverse_source = np.zeros_like(sorted_source)
    reverse_source[np.arange(0, n)[:, None], sorted_source] = np.arange(0, source.shape[0])[None, :]
    sorted_target = np.argsort(target_projections, axis=-1)

    proj_diffs = target_projections[np.arange(n)[:, None], sorted_target[np.arange(n)[:, None], reverse_source]] - \
                 source_projections
    swd = np.mean(np.square(proj_diffs))

    if grad:
        return swd, np.einsum("ij,id->jd", proj_diffs, directions) / n
    else:
        return swd


class SamplingWassersteinInterpolation:

    def __init__(self, init_samples, target_sampler, perf_lb, epsilon, bounds=[0.0, 1.0], callback=None):
        self.current_samples = init_samples
        self.n_samples, self.dim = self.current_samples.shape
        self.target_sampler = target_sampler
        self.bounds = bounds
        self.perf_lb = perf_lb
        self.epsilon = epsilon
        self.callback = callback

    def sample_ball(self, targets, samples=None, half_ball=None, n=100):
        if samples is None:
            samples = self.current_samples

        # Taken from http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
        # Method 20
        direction = np.random.normal(0, 1, (n, self.dim))
        norm = np.linalg.norm(direction, axis=-1, keepdims=True)
        r = np.power(np.random.uniform(size=(n, 1)), 1. / self.dim)

        # We only consider samples that decrease the distance objective (i.e. are aligned with the direction)
        noise = r * (direction / norm)
        dirs = targets - samples
        dir_norms = np.einsum("ij,ij->i", dirs, dirs)
        noise_projections = np.einsum("ij,kj->ik", dirs / dir_norms[:, None], noise)

        projected_noise = np.where((noise_projections > 0)[..., None], noise[None, ...],
                                   noise[None, ...] - 2 * noise_projections[..., None] * dirs[:, None, :])
        if half_ball is not None:
            projected_noise[~half_ball] = noise

        scales = np.minimum(self.epsilon, np.sqrt(dir_norms))[:, None, None]
        return np.clip(samples[..., None, :] + scales * projected_noise, self.bounds[0], self.bounds[1])

    @staticmethod
    def visualize_particles(init_samples, particles, performances):
        if particles.shape[-1] != 2:
            raise RuntimeError("Can only visualize 2D data")

        import matplotlib.pyplot as plt
        f = plt.figure()
        ax = f.gca()
        scat = ax.scatter(particles[0, :, 0], particles[0, :, 1], c=performances[0, :])
        ax.scatter(init_samples[0, 0], init_samples[0, 1], marker="x", c="red")
        plt.colorbar(scat)
        plt.show()

    def ensure_successful_initial(self, model, init_samples, success_samples):
        performance_reached = model.predict_individual(init_samples) >= self.perf_lb
        replacement_mask = ~performance_reached
        n_replacements = np.sum(replacement_mask)
        if n_replacements > 0:
            valid_successes = model.predict_individual(success_samples) >= self.perf_lb
            n_valid = np.sum(valid_successes)
            if n_valid >= n_replacements:
                # In this case we only allow selection from the valid samples
                success_samples = success_samples[valid_successes, :]
                valid_successes = np.ones(success_samples.shape[0], dtype=bool)

            dists = np.sum(np.square(success_samples[None, :, :] - init_samples[replacement_mask, None, :]), axis=-1)
            success_assignment = linear_sum_assignment(dists, maximize=False)
            init_samples[replacement_mask, :] = success_samples[success_assignment[1], :]
            performance_reached[replacement_mask] = valid_successes[success_assignment[1]]

        return init_samples, performance_reached

    def update_distribution(self, model, success_samples, debug=False):
        init_samples, performance_reached = self.ensure_successful_initial(model, self.current_samples.copy(),
                                                                           success_samples)
        target_samples = self.target_sampler(self.n_samples)
        if debug:
            target_samples_true = target_samples.copy()
        movements = sliced_wasserstein(init_samples, target_samples, grad=True)[1]
        target_samples = init_samples + movements
        particles = self.sample_ball(target_samples, samples=init_samples, half_ball=performance_reached)

        distances = np.linalg.norm(particles - target_samples[:, None, :], axis=-1)
        performances = model.predict_individual(particles)
        if debug:
            self.visualize_particles(init_samples, particles, performances)

        mask = performances > self.perf_lb
        solution_possible = np.any(mask, axis=-1)
        distances[~mask] = np.inf
        opt_idxs = np.where(solution_possible, np.argmin(distances, axis=-1), np.argmax(performances, axis=-1))
        new_samples = particles[np.arange(0, self.n_samples), opt_idxs]

        if debug:
            vis_idxs = np.random.randint(0, target_samples.shape[0], size=50)
            import matplotlib.pyplot as plt
            xs, ys = np.meshgrid(np.linspace(0, 9, num=150), np.linspace(0, 6, num=100))
            zs = model.predict_individual(np.stack((xs, ys), axis=-1))
            ims = plt.imshow(zs, extent=[0, 9, 0, 6], origin="lower")
            plt.contour(xs, ys, zs, [180])
            plt.colorbar(ims)

            plt.scatter(target_samples_true[vis_idxs, 0], target_samples_true[vis_idxs, 1], marker="x", color="red")
            plt.scatter(self.current_samples[vis_idxs, 0], self.current_samples[vis_idxs, 1], marker="o", color="C0")
            plt.scatter(init_samples[vis_idxs, 0], init_samples[vis_idxs, 1], marker="o", color="C2")
            plt.scatter(new_samples[vis_idxs, 0], new_samples[vis_idxs, 1], marker="o", color="C1")
            plt.xlim([0, 9])
            plt.ylim([0, 6])
            plt.show()

        if self.callback is not None:
            self.callback(self.current_samples, new_samples, success_samples, target_samples)

        self.current_samples = new_samples

    def save(self, path):
        with open(os.path.join(path, "teacher.pkl"), "wb") as f:
            pickle.dump((self.current_samples, self.perf_lb, self.epsilon), f)

    def load(self, path):
        with open(os.path.join(path, "teacher.pkl"), "rb") as f:
            tmp = pickle.load(f)

            self.current_samples = tmp[0]
            self.n_samples = self.current_samples.shape[0]

            self.perf_lb = tmp[1]
            self.epsilon = tmp[2]
        

class WassersteinSuccessBuffer:

    def __init__(self, delta: float, n: int, epsilon: float, context_bounds: Tuple[np.ndarray, np.ndarray]):
        context_exts = context_bounds[1] - context_bounds[0]
        self.delta_stds = context_exts / 4
        self.min_stds = 0.005 * epsilon * np.ones(len(context_bounds[0]))
        self.context_bounds = context_bounds
        self.delta = delta
        self.max_size = n
        self.contexts = np.zeros((1, len(context_bounds[0])))
        self.returns = np.array([-np.inf])
        self.delta_reached = False
        self.min_ret = None

    def update(self, contexts, returns, current_samples):
        assert contexts.shape[0] < self.max_size

        if self.min_ret is None:
            self.min_ret = np.min(returns)

        if not self.delta_reached:
            self.delta_reached, self.contexts, self.returns, mask = self.update_delta_not_reached(contexts, returns, current_samples)
        else:
            self.contexts, self.returns, mask = self.update_delta_reached(contexts, returns, current_samples)

        return contexts[mask, :], returns[mask]

    def update_delta_not_reached(self, contexts: np.ndarray, returns: np.ndarray, current_samples: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray, List[bool]]:
        # Only add samples that have a higher return than the median return in the buffer (we do >= here to allow
        # for binary rewards to work)
        med_idx = self.returns.shape[0] // 2
        mask = returns >= self.returns[med_idx]
        n_new = np.sum(mask)
        print("Improving buffer quality with %d samples" % n_new)

        # We do not want to shrink the buffer
        offset_idx = med_idx + 1
        if n_new < offset_idx:
            offset_idx = n_new

        new_returns = np.concatenate((returns[mask], self.returns[offset_idx:]), axis=0)
        new_contexts = np.concatenate((contexts[mask, :], self.contexts[offset_idx:, :]), axis=0)
        sort_idxs = np.argsort(new_returns)

        # Ensure that the buffer is only growing, never shrinking and that all the buffer sizes are consistent
        assert self.contexts.shape[0] <= new_contexts.shape[0]
        assert new_contexts.shape[0] == new_returns.shape[0]

        # These are the indices of the tasks that have NOT been added to the buffer (so the negation of the mas)
        rem_mask = ~mask

        # Ensure that we are not larger than the maximum size
        if new_returns.shape[0] > self.max_size:
            sort_idxs = sort_idxs[-self.max_size:]
            # Since we are clipping potentially removing some of the data chunks we need to update the remainder mask
            # Since we add the new samples at the beginning of the new buffers, we are interested whether the idxs
            # in [0, n_new) are still in the sort_idxs array. If this is NOT the case, then the sample has NOT been
            # added to the buffer.
            rem_mask[mask] = [i not in sort_idxs for i in np.arange(n_new)]

        new_delta_reached = self.returns[self.returns.shape[0] // 2] > self.delta
        return new_delta_reached, new_contexts[sort_idxs, :], new_returns[sort_idxs], rem_mask

    def update_delta_reached(self, contexts: np.ndarray, returns: np.ndarray, current_samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[bool]]:
        # Compute the new successful samples
        mask = returns >= self.delta
        n_new = np.sum(mask)

        if n_new > 0:
            remove_mask = self.returns < self.delta
            if not np.any(remove_mask) and self.returns.shape[0] >= self.max_size:
                extended_contexts = np.concatenate((self.contexts, contexts[mask, :]), axis=0)
                extended_returns = np.concatenate((self.returns, returns[mask]), axis=0)

                # At this stage we use the optimizer
                dists = np.sum(np.square(extended_contexts[:, None, :] - current_samples[None, :, :]), axis=-1)
                assignments = linear_sum_assignment(dists, maximize=False)
                ret_idxs = assignments[0]

                # Select the contexts using the solution from the MIP solver. The unique functions sorts the data
                new_contexts = extended_contexts[ret_idxs, :]
                new_returns = extended_returns[ret_idxs]

                # We update the mask to indicate only the kept samples
                mask[mask] = [idx in (ret_idxs - self.contexts.shape[0]) for idx in np.arange(n_new)]

                avg_dist = sliced_wasserstein(new_contexts, current_samples)
                print("Updated success buffer with %d samples. New Wasserstein distance: %.3e" % (n_new, avg_dist))
            else:
                # We replace the unsuccessful samples by the successful ones
                if n_new < np.sum(remove_mask):
                    remove_idxs = np.argpartition(self.returns, kth=n_new)[:n_new]
                    remove_mask = np.zeros(self.returns.shape[0], dtype=bool)
                    remove_mask[remove_idxs] = True

                new_returns = np.concatenate((returns[mask], self.returns[~remove_mask]), axis=0)
                new_contexts = np.concatenate((contexts[mask, :], self.contexts[~remove_mask, :]), axis=0)

                if new_returns.shape[0] > self.max_size:
                    new_returns = new_returns[:self.max_size]
                    new_contexts = new_contexts[:self.max_size, :]

                # Ensure that the buffer is only growing, never shrinking and that all the buffer sizes are consistent
                assert self.contexts.shape[0] <= new_contexts.shape[0]
                assert new_contexts.shape[0] == new_returns.shape[0]
        else:
            new_contexts = self.contexts
            new_returns = self.returns

        return new_contexts, new_returns, ~mask

    def read_train(self):
        return self.contexts.copy(), self.returns.copy()

    def read_update(self):
        # Compute the Gaussian search noise that we add to the samples
        var_scales = np.clip(self.delta - self.returns, 0., np.inf) / (self.delta - self.min_ret)
        stds = self.min_stds[None, :] + var_scales[:, None] * self.delta_stds[None, :]

        # If we did not yet reach the desired threshold we enforce exploration by scaling the exploration noise w.r.t.
        # the distance to the desired threshold value
        if not self.delta_reached:
            offset = self.returns.shape[0] // 2
            sub_returns = self.returns[offset:]
            sub_contexts = self.contexts[offset:, :]
            sub_stds = stds[offset:, :]

            # Do a resampling based on the achieved rewards (favouring higher rewards to be resampled)
            probs = sub_returns - self.returns[offset - 1]
            norm = np.sum(probs)
            if norm == 0:
                probs = np.ones(sub_returns.shape[0]) / sub_returns.shape[0]
            else:
                probs = probs / norm

            sample_idxs = np.random.choice(sub_returns.shape[0], self.max_size, p=probs)
            sampled_contexts = sub_contexts[sample_idxs, :]
            sampled_stds = sub_stds[sample_idxs, :]
        else:
            to_fill = self.max_size - self.returns.shape[0]
            add_idxs = np.random.choice(self.returns.shape[0], to_fill)
            sampled_contexts = np.concatenate((self.contexts, self.contexts[add_idxs, :]), axis=0)
            sampled_stds = np.concatenate((stds, stds[add_idxs, :]), axis=0)

        contexts = sampled_contexts + np.random.normal(0, sampled_stds, size=(self.max_size, self.contexts.shape[1]))
        invalid = np.any(np.logical_or(contexts < self.context_bounds[0][None, :],
                                       contexts > self.context_bounds[1][None, :]), axis=-1)
        count = 0
        while np.any(invalid) and count < 10:
            new_noise = np.random.normal(0, sampled_stds[invalid, :], size=(np.sum(invalid), self.contexts.shape[1]))
            contexts[invalid, :] = sampled_contexts[invalid, :] + new_noise
            invalid = np.any(np.logical_or(contexts < self.context_bounds[0][None, :],
                                           contexts > self.context_bounds[1][None, :]), axis=-1)
            count += 1

        return np.clip(contexts, self.context_bounds[0], self.context_bounds[1])


class NadarayaWatson:

    def __init__(self, contexts, returns, lengthscale=None, n_threads=5, n_max=None, radius_scale=3.):
        self.model = na.NadarayaWatson(contexts, returns, n_threads=n_threads)
        if lengthscale is None:
            self.lengthscale = np.median(pdist(contexts))
        else:
            self.lengthscale = lengthscale

        if n_max is None:
            self.n_max = int(0.5 * contexts.shape[0])
        else:
            self.n_max = n_max

        self.radius_scale = radius_scale

    def predict_individual(self, x):
        return np.reshape(self.model.predict(np.reshape(x, (-1, x.shape[-1])), self.lengthscale, n_max=self.n_max,
                                             radius_scale=self.radius_scale), x.shape[:-1])

    def save(self, path):
        pass

    def load(self, path):
        pass


class CurrOT(Scheduler):
    
    def __init__(self, reward_dim, update_frequency, 
                 tau=20,
                 seed=None, 
                 init_sample_size=128, 
                 success_threshold=5.0, 
                 epsilon=1.5,
                 update_weights_frequency=1,
                 sampling_strategy="past"):
        # delta is environment specific. since max reward in panda experiments is ~45 or ~95 -- depending on the episode lenght -- we set the default performance threshold to 5.
        super(CurrOT, self).__init__(reward_dim=reward_dim, tau=tau, seed=seed, update_frequency=update_frequency)
        
        self.threshold_reached = False
        self.wait_until_threshold = False
        
        init_samples = self.rng.random((init_sample_size, reward_dim)) * 1
        
        target_mu = th.ones((reward_dim, )) * 0
        target_mu[-1] = 1
        
        target_sampler = lambda n_samples : self.rng.normal(loc=target_mu, scale=0.1, size=(n_samples, self.reward_dim))
        # target_dist = torchrl.modules.TruncatedNormal(target_mu, th.tensor(0.1), upscale=1.0, min=0.0, max=1.0, tanh_loc=True)
        # target_sampler = lambda n_samples : target_dist.sample((n_samples, )).numpy()
        
        self.teacher = SamplingWassersteinInterpolation(init_samples, target_sampler, success_threshold, np.sqrt(epsilon))
        
        self.success_buffer = WassersteinSuccessBuffer(success_threshold, init_samples.shape[0], epsilon, (np.zeros(reward_dim), np.ones(reward_dim) * 1))
        self.fail_context_buffer = []
        self.fail_return_buffer = []
        
        assert sampling_strategy in {"past", "current", "random"}
        self.sampling_strategy = sampling_strategy
        self.update_weights_frequency = update_weights_frequency
        self.weight_buffer = []
        self.main_weight_buffer = []
        self.weighted_rewards_buffer = []
        self.main_rewards_buffer = []
        
        if self.sampling_strategy == "past":
            self.all_weights_history = []
        
        self.n_maybe_update = 0
        self.n_updates = 0
        
    def init_period(self):
        self.weighted_rewards_buffer = []
        self.weight_buffer = []
            
        return super().init_period()
        
    def maybe_update(self, **kwargs):
        assert "episode_rewards" in kwargs
        
        weighted_reward = (kwargs["episode_rewards"].reshape(1, -1) @ self.get_current_weights().reshape(-1, 1)).item()
        self.weighted_rewards_buffer.append(weighted_reward)
        self.weight_buffer.append(self.current_weights)
            
        if (self.n_maybe_update + 1) % self.update_weights_frequency == 0:
            self.main_rewards_buffer.append(np.mean(self.weighted_rewards_buffer))
            self.main_weight_buffer.append(np.stack(self.weight_buffer, axis=0).mean(axis=0))
            
            self.init_period()
            
        if (self.n_maybe_update + 1) % self.update_frequency == 0:
            # do the update
            self.update()
            self.n_updates += 1
            
            self.main_rewards_buffer = []
            self.main_weight_buffer = []
            
            was_curriculum_updated = True
        else:
            was_curriculum_updated = False
        self.n_maybe_update += 1
        
        return was_curriculum_updated
    
    def update(self):
        weight_buffer = np.stack(self.main_weight_buffer, axis=0)
        reward_buffer = np.stack(self.main_rewards_buffer, axis=0)
        
        fail_contexts, fail_returns = self.success_buffer.update(weight_buffer, reward_buffer, self.teacher.target_sampler(self.teacher.current_samples.shape[0]))

        if self.threshold_reached:
            self.fail_context_buffer.extend(fail_contexts)
            self.fail_context_buffer = self.fail_context_buffer[-self.teacher.n_samples:]
            self.fail_return_buffer.extend(fail_returns)
            self.fail_return_buffer = self.fail_return_buffer[-self.teacher.n_samples:]

        success_contexts, success_returns = self.success_buffer.read_train()
        if len(self.fail_context_buffer) == 0:
            train_contexts = success_contexts
            train_returns = success_returns
        else:
            train_contexts = np.concatenate((np.stack(self.fail_context_buffer, axis=0), success_contexts), axis=0)
            train_returns = np.concatenate((np.stack(self.fail_return_buffer, axis=0), success_returns), axis=0)
        model = NadarayaWatson(train_contexts, train_returns, 0.3 * self.teacher.epsilon)
        
        avg_perf = np.mean(model.predict_individual(self.teacher.current_samples))
        if self.threshold_reached or avg_perf >= self.teacher.perf_lb:
            self.threshold_reached = True
            self.teacher.update_distribution(model, self.success_buffer.read_update())
        else:
            print("Current performance: %.3e vs %.3e" % (avg_perf, self.teacher.perf_lb))
            if self.wait_until_threshold:
                print("Not updating sampling distribution, as performance threshold not met")
            else:
                # Which update is better is better?
                self.teacher.update_distribution(model, self.success_buffer.read_update())
                # self.teacher.current_samples = self.success_buffer.read_update()
        
    def get_current_weights(self):
        return self.current_weights # / self.current_weights.sum()
        current_weights = self.current_weights * self.tau
        e_weights = np.exp(current_weights - current_weights.max())
        return e_weights / e_weights.sum()
    
    def sample(self):
        sample_idx = self.rng.integers(0, len(self.teacher.current_samples))
        self.current_weights = self.teacher.current_samples[sample_idx, :]
        
        if self.sampling_strategy == "past":
            self.all_weights_history.append(self.current_weights)
        
    def sample_batch(self, batch_size):
        if self.sampling_strategy == "current":
            sample_idxs = self.rng.integers(low=0, high=len(self.teacher.current_samples), size=batch_size)
            weight_batch = self.teacher.current_samples[sample_idxs, :]
        elif self.sampling_strategy == "past":
            sample_idxs = self.rng.integers(low=0, high=len(self.all_weights_history), size=batch_size)
            weight_batch = np.array([self.all_weights_history[idx] for idx in sample_idxs])
        elif self.sampling_strategy == "random":
            weight_batch = self.rng.random((batch_size, self.reward_dim))
        return weight_batch # / weight_batch.sum(axis=1, keepdims=True)
        
        weight_batch = weight_batch * self.tau
        e_weight_batch = np.exp(weight_batch - weight_batch.max())
        return e_weight_batch / e_weight_batch.sum(axis=1, keepdims=True)
