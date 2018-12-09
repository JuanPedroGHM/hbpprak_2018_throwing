import numpy as np
import gym
from evostra.algorithms.evolution_strategy import EvolutionStrategy
from evostra.models.feed_forward_network import FeedForwardNetwork

import pickle

#global vars
rewards = []
n_threads = 1
render = True
pop_size = 50
pop_index = 0


def update_pop_index():
    global pop_index
    pop_index = (pop_index + 1) % pop_size

def get_reward(weights):
    global observations, model, envs, pop_index
    model.set_weights(weights)
    prediction = model.predict(observations[pop_index])
    sm_output = np.exp(prediction)/np.sum(np.exp(prediction))
    decision = np.argmax(sm_output)
    observations[pop_index], reward, done, info = envs[pop_index].step(decision)
    update_pop_index()
    if done:
        observation = envs[pop_index].reset()
        update_pop_index()
        return 0

    return reward

environment = 'MountainCar-v0'

envs = [gym.make(environment) for i in range(pop_size)]
observations = [env.reset() for env in envs]

n_actions = envs[0].action_space.n
n_observations = len(observations[0])
topology = [n_observations,8,6,n_actions]

model = FeedForwardNetwork(topology)

es = EvolutionStrategy(model.get_weights(), get_reward, population_size=pop_size, sigma=0.2, learning_rate=0.03,
                              decay=0.995, num_threads=n_threads)

es.run(100000*pop_size, print_step=1000)

#save the optimal model
optimized_weights = es.get_weights()
model.set_weights(optimized_weights)


with open(environment+"-"+str(pop_size)+"-"+str(topology)+".pickle","wb") as f:
    pickle.dump(model,f,pickle.HIGHEST_PROTOCOL)
