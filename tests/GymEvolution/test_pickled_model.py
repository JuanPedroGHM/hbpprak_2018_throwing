import gym
from os import path
import pickle
import numpy as np
import matplotlib.pyplot as plt

from evostra.algorithms.evolution_strategy import EvolutionStrategy
from evostra.models.feed_forward_network import FeedForwardNetwork


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
    gym_env = 'MountainCar-v0'
    testenv = gym.make(gym_env)
    model = load_model('MountainCar-v0-50-[2, 8, 6, 3].pickle')
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
    main()