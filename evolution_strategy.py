from __future__ import print_function
import numpy as np
import multiprocessing as mp

class EvolutionStrategy(object):
    def __init__(self, topology, weights, bias, get_reward_func, population_size=50, sigma=0.1, learning_rate=0.03, decay=0.999,
                 num_threads=1):

        self.topology = topology
        self.weights = weights
        self.bias = bias
        self.get_reward = get_reward_func
        self.POPULATION_SIZE = population_size
        self.SIGMA = sigma
        self.learning_rate = learning_rate
        self.decay = decay
        self.num_threads = mp.cpu_count() if num_threads == -1 else num_threads

    def _get_weights_try(self, w, p):
        weights_try = []
        for index, i in enumerate(p):
            jittered = self.SIGMA * i
            weights_try.append(w[index] + jittered)
        return weights_try

    def get_weights(self):
        return self.weights

    def _get_population(self):
        population = []
        for i in range(self.POPULATION_SIZE):
            x = []
            ind_bias = []
            for w in self.weights:
                x.append(np.random.randn(*w.shape))
                ind_bias.append(np.random.randn(1,w.shape[1]))
            population.append((x,ind_bias))
        return population

    def _get_rewards(self, pool, population):
        if pool is not None:
            worker_args = ((self.get_reward, self._get_weights_try(self.weights, p), self.topology) for p in population)
            rewards = pool.map(worker_process, worker_args)

        else:
            rewards = []
            for index, p in enumerate(population):
                weights_try = self._get_weights_try(self.weights, p[0])
                bias_try = self._get_weights_try(self.bias, p[1])
                rewards.append(self.get_reward(index, weights_try, bias_try, self.topology))
                print("Finished with inididual {}".format(index))
        rewards = np.array(rewards)
        return rewards

    def _update_weights(self, rewards, population):
        std = rewards.std()
        if std == 0:
            return 
        rewards = (rewards - rewards.mean()) / std
        print("Updating weights")
        for index, w in enumerate(self.weights):
            layer_weight_population = np.array([p[0][index] for p in population])
            layer_bias_population = np.array([p[1][index] for p in population])

            update_factor = self.learning_rate / (self.POPULATION_SIZE * self.SIGMA)
            self.weights[index] = w + update_factor * np.dot(layer_weight_population.T, rewards).T
            self.bias[index] = self.bias[index] + update_factor * np.dot(layer_bias_population.T, rewards).T
        self.learning_rate *= self.decay

    def run(self, iterations, print_step=10):
        pool = mp.Pool(self.num_threads) if self.num_threads > 1 else None
        average_reward = []
        for iteration in range(iterations):

            population = self._get_population()
            rewards = self._get_rewards(pool, population)

            self._update_weights(rewards, population)
            average_reward.append(rewards.mean())

            if (iteration + 1) % print_step == 0:
                print('iter %d. reward: %f' % (iteration + 1, rewards.mean()))
        if pool is not None:
            pool.close()

        print("AVG reward:" + str(average_reward))
        return average_reward

