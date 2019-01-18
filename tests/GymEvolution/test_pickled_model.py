import gym
from os import path
import pickle
import numpy as np
import matplotlib.pyplot as plt

from evostra.algorithms.evolution_strategy import EvolutionStrategy
from evostra.models.feed_forward_network import FeedForwardNetwork
#from evostra.models.spiking_feed_forward_network import Spiking_feed_forward_network


def load_model(name):
    if not path.isfile(name):
        name = "tests/GymEvolution/"+name
    try:
        with open(name,"rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError as e:
        print(e)

def main():
    gym_env = 'CartPole-v1'
    #gym_env = 'MountainCar-v0'
    testenv = gym.make(gym_env)
    #model = load_model('MountainCar-v0-50-[2, 10, 6, 3].pickle')
    model = load_model('CartPole-v1-30-[4, 10, 6, 2]_SNN.pickle')
    total_reward = []
    high_obs = testenv.observation_space.high
    low_obs = testenv.observation_space.low


    for t in range(10):
        done = False
        observation = testenv.reset()
        rewards = []
        while not done:
            testenv.render()
            prediction = model.predict(observation,high_obs,low_obs)
            #sm = np.exp(prediction) / np.sum(np.exp(prediction))
            decision = np.argmax(prediction)
            #decision = testenv.action_space.sample()
            observation, reward, done, info = testenv.step(decision)
            rewards.append(reward)
            if done:
                total_reward.append(np.sum(rewards))
                print("Episode finished after {} timesteps".format(t + 1))
    plt.plot(total_reward)
    plt.show()


if __name__ == '__main__':
    main()