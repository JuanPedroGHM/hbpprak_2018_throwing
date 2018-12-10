import numpy as np
import gym
from evostra.algorithms.evolution_strategy import EvolutionStrategy
from evostra.models.feed_forward_network import FeedForwardNetwork
from evostra.models.spiking_feed_forward_network import Spiking_feed_forward_network


import warnings

import pickle
import matplotlib.pyplot as plt

#global vars

n_threads = 2
pop_size = 50
learning_rate = 0.03
decay = 0.99
sigma = 0.2
environment = 'CartPole-v1'

hiddenLayers = [10,6]


def get_reward(weights,env_name,topology):

    model = Spiking_feed_forward_network(topology)
    model.set_weights(weights)

    env = gym.make(env_name)
    observation = env.reset()
    high_obs = env.observation_space.high
    low_obs = env.observation_space.low
    rewards = []
    done = False

    while (not done):
        prediction = model.predict(observation,high_obs,low_obs)
        sm = np.exp(prediction) / np.sum(np.exp(prediction))
        decision = np.argmax(sm)
        observation, reward, done, info = env.step(decision)
        rewards.append(reward)
    return np.sum(rewards)

def check_lr_dep_plausability(lr,dep,iterations):
    course = [lr*dep**iter for iter in range(iterations)]
    plt.plot(course)
    plt.show()

    if course[len(course)-1]>0.005:
        print("Decay may not be high enough")
    if course[len(course)-1]>0.1*course[0]:
        print("Decay may not be high enough")
    if course[int(len(course)*0.1)]<0.7*learning_rate:
        print("Decay may be too high")

def test_model(model_name,model):

    testenv = gym.make(model_name)

    testenv = gym.make(model_name)
    total_reward = []
    for i_episode in range(20):
        observation = testenv.reset()
        rewards = []
        for t in range(10000):
            testenv.render()
            prediction = model.predict(observation)
            sm = np.exp(prediction) / np.sum(np.exp(prediction))
            decision = np.argmax(sm)
            decision = testenv.action_space.sample()
            observation, reward, done, info = testenv.step(decision)
            rewards.append(reward)
            if done:
                total_reward.append(np.sum(rewards))
                print("Episode finished after {} timesteps".format(t + 1))
                break
    plt.plot(total_reward)
    plt.show()

if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    test_env = gym.make(environment)

    n_actions = test_env.action_space.n
    n_observations = len(test_env.reset())
    topology = [n_observations]
    topology.extend(hiddenLayers)
    topology.append(n_actions)

    iterations = 10*pop_size

    check_lr_dep_plausability(learning_rate,decay,iterations)

    input("Continue?")

    model = FeedForwardNetwork(topology)

    es = EvolutionStrategy(topology, model.get_weights(), get_reward, population_size=pop_size, sigma=sigma, learning_rate=learning_rate,
                           decay=decay, num_threads=n_threads,env_name=environment)

    es.run(iterations, print_step=10)

    #save the optimal model
    optimized_weights = es.get_weights()
    model.set_weights(optimized_weights)


    with open(environment+"-"+str(pop_size)+"-"+str(topology)+".pickle","wb") as f:
        pickle.dump(model,f,pickle.HIGHEST_PROTOCOL)

    input("Enter to go for test")
    test_model(environment,model)
