# %% import pyNN.nest as sim
from pyNN.random import RandomDistribution as rnd
from pyNN.utility import get_simulator, normalized_filename
from pyNN.utility.plotting import Figure, Panel
import numpy as np
import matplotlib.pyplot as plt
import gym
from functools import reduce
import quantities as units

## Configuration params
inputNeuronFactor = 20
outputNeuronFactor = 10
env_name = 'MountainCar-v0'
step_size = 10.0
render = True
trials = 200

## Training params
Apos = 1
Aneg = -0.5
tauPos = 20
tauNeg = 20
tauTrace = 40

learningRate = 1

## Helper functions and classes
def getAction():
    pass

def getRate(value, minV, maxV):
    minRate = 0.0
    maxRate = 50.0
    return (value - minV)/(maxV - minV) * (maxRate - minRate) + minRate

def plot_spiketrains(inputPop, outputPop, filename):
    Figure(
        Panel(inputPop.get_data().segments[0].spiketrains, ylabel='Input Neuron Index', xticks=True),
        Panel(outputPop.get_data().segments[0].spiketrains, ylabel='Output Neuron Index', xticks=True),
        Panel(outputPop.get_data().segments[0].filter(name='v')[0], ylabel='Output Membrane Potential', xlabel="Time (ms)", xticks=True),
        title="Helper plots"
    ).save(normalized_filename(filename, "spikes", "png", options.simulator))


class SpikeTraceTracker():

    def __init__(self, shape, inputPop, outputPop, sampling_rate = 1.0):
        self.shape = shape
        self.sampling_rate = sampling_rate
        self.inputPop = inputPop
        self.outputPop = outputPop
        self.segment = 0
        self.reset()

    def __call__(self, t):

        deltaTrace = np.zeros(self.shape)
        inputSpikes, outputSpikes = [], []
        try:
            inputSpikes = self.inputPop.get_data().segments[0].spiketrains
            outputSpikes = self.outputPop.get_data().segments[0].spiketrains
        except:
            return t + self.sampling_rate

        for j in range(len(inputSpikes)):
            if self.neuronFired(inputSpikes[j], t):
                self.Pplus[j] += Apos
                deltaTrace[j,:] += self.Pminus
            else:
                self.Pplus[j] *= np.exp(-self.sampling_rate / tauPos)

        for i in range(len(outputSpikes)):
            if self.neuronFired(outputSpikes[i], t):
                self.Pminus[i] += Aneg
                deltaTrace[:,i] += self.Pplus
                self.spikeCount[i] += 1
            else:
                self.Pminus[i] *= np.exp(-self.sampling_rate / tauNeg)

        self.eTrace = self.eTrace * np.exp(-self.sampling_rate / tauTrace) + deltaTrace

        return t + self.sampling_rate

    def neuronFired(self, neuronSpiketrain, t):
        if len(neuronSpiketrain) > 0:
            return neuronSpiketrain[-1] >= (t - self.sampling_rate)
        else:
            return False

    def reset(self):
        self.Pplus = np.zeros((self.shape[0], ))
        self.Pminus = np.zeros((self.shape[1], ))
        self.eTrace = np.zeros(self.shape)
        self.spikeCount = np.zeros((self.shape[1],))

    def resetCount(self):
        self.spikeCount = np.zeros((self.shape[1],))

    def setSegmentIndex(self, index):
        self.segment = index

##
env = gym.make(env_name)
env.reset()

nActions = env.action_space.n
nObservations = env.observation_space.shape[0]

# %% Neuron params
sim, options = get_simulator(("--plot-figure", "Plot the simulation results to a file.",
                              {"action": "store_true"}))

sim.setup(timestep = 1.0)

params = {
    'a' : 0.02,
    'b' : 0.2,
    'c' : rnd('uniform', [-65, -50]),
    'd' : rnd('uniform', [2, 8])
}

inputPop = sim.Population(nObservations * inputNeuronFactor, sim.SpikeSourcePoisson())
outputPop = sim.Population(nActions * outputNeuronFactor, sim.IF_curr_exp())
syn = sim.StaticSynapse(weight=rnd('uniform', [0, 3]))
connections = sim.Projection(inputPop, outputPop, sim.AllToAllConnector(), syn)

weights = connections.get('weight', format='array')

inputPop.record('spikes')
outputPop.record(['spikes', 'v'])


spikeTraceTracker = SpikeTraceTracker(weights.shape, inputPop, outputPop)

for i in range(trials):

    done = False
    observation = env.reset()
    totalReward = 0.0
    totalAdjustedReward = 0.0
    spikeTraceTracker.setSegmentIndex(i)
    lastSpeed = 0.0

    maxSpeed = 0.0

    while(not done):

        if render:
            env.render()

        ## feed env output to net

        for value, minV, maxV, index in zip(observation,
                                        env.observation_space.low,
                                        env.observation_space.high,
                                        list(range(inputNeuronFactor))):
            inputPop[index*inputNeuronFactor:(index+1)*inputNeuronFactor].set(rate=getRate(value, minV, maxV ))

        # Rum network simulation
        sim.run(step_size, callbacks=[spikeTraceTracker])

        ## read net output for action
        # Divide the neurons equally between all the actions, the group of neurons that fire the most
        # in the last time step, is the action

        # There must be a better way to do this
        actionCount = [x.sum() for x in np.split(spikeTraceTracker.spikeCount, nActions)]
        spikeTraceTracker.resetCount()

        action = 0
        if np.all(actionCount == np.max(actionCount)):
            action = np.random.choice(nActions)
        else:
            action = np.argmax(actionCount)

        ## take env step
        observation, reward, done, info = env.step(action)
        totalReward += reward

        reward = reward if reward > 0 else 0
        reward += np.abs(observation[1]) - lastSpeed
        totalAdjustedReward += reward

        lastSpeed = np.abs(observation[1])
        maxSpeed = np.max([lastSpeed, maxSpeed])
        print(f'Action Count: {actionCount}, Action #: {action}, Reward: {reward}, Speed: {lastSpeed}' )

        ## perform learning step
        weights += learningRate * reward * spikeTraceTracker.eTrace
        connections.set(weight = weights)


    ## Finished simulation
    print(f'Trial {i}, Total env reward = {totalReward}, Total adjusted reward: {totalAdjustedReward}, Max speed: {maxSpeed}' )

plot_spiketrains(inputPop, outputPop, 'results')
sim.end()
print('Simulation ended')