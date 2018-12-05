# %% import pyNN.nest as sim
from pyNN.random import RandomDistribution as rnd
from pyNN.utility import get_simulator, normalized_filename
from pyNN.utility.plotting import Figure, Panel
import numpy as np
import matplotlib.pyplot as plt
import gym

# %% Constants and config params
inputNeuronFactor = 20
outputNeuronFactor = 20
env_name = 'MountainCar-v0'
step_size = 50.0
render = True
trials = 100

# %% Helper Functions
def getAction():
    pass

def getRate(value, minV, maxV):
    minRate = 0.0
    maxRate = 1/(step_size /1000)
    return (value - minV)/(maxV - minV) * (maxRate - minRate) + minRate

def plot_spiketrains(inputPop, outputPop, filename):
    Figure(
        Panel(inputPop.get_data().segments[0].spiketrains, ylabel='Input Neuron Index', xticks=True),
        Panel(outputPop.get_data().segments[0].spiketrains, ylabel='Output Neuron Index', xticks=True),
        Panel(outputPop.get_data().segments[0].filter(name='v')[0], ylabel='Output Membrane Potential', xlabel="Time (ms)", xticks=True),
        title="Helper plots"
    ).save(normalized_filename(filename, "spikes", "png", options.simulator))

# %% setup env
env = gym.make(env_name)
env.reset()

nActions = env.action_space.n
nObservations = env.observation_space.shape[0]

# %% Neuron params
sim, options = get_simulator(("--plot-figure", "Plot the simulation results to a file.",
                              {"action": "store_true"}))

sim.setup()

params = {
    'a' : 0.02,
    'b' : 0.2,
    'c' : rnd('uniform', [-65, -50]),
    'd' : rnd('uniform', [2, 8])
}

inputPop = sim.Population(nObservations * inputNeuronFactor, sim.SpikeSourcePoisson())
outputPop = sim.Population(nActions * outputNeuronFactor, sim.Izhikevich(**params))
syn = sim.StaticSynapse(weight=rnd('uniform', [0, 5]))
connections = sim.Projection(inputPop, outputPop, sim.AllToAllConnector(), syn)

inputPop.record('spikes')
outputPop.record(['spikes', 'v'])

for i in range(trials):

    done = True
    observation = []
    totalReward = 0

    while(True):

        if done:
            obsevation = env.reset()

        if render:
            env.render()

            ## feed env output to net
        for value, minV, maxV, index in zip(observation,
                                        env.observation_space.low,
                                        env.observation_space.high,
                                        list(range(inputNeuronFactor))):
            inputPop[index*inputNeuronFactor:(index+1)*inputNeuronFactor].set(rate=getRate(value, minV, maxV ))

        # Rum network simulation
        sim.run(step_size)

        ## read net output for action
        action = env.action_space.sample()
        
        ## take env step
        observation, reward, done, info = env.step(action)
        totalReward += reward

        ## perform learning step
        

# plot_spiketrains(inputPop, outputPop, 'results')
