import numpy as np
import pyNN.nest as sim
from pyNN import random
from pyNN.utility import get_simulator


#topologiy variables
input_pop = 3
output_pop = 2

reg_spikingR = 60
adapitveA = 20

#cell params

fiering_period = 100
params = {
    "tau_m":2,
    "v_tres": -50,
    "v_res": -50,
    "tau_refac":fiering_period/2

}

#simulation params
timestep = 2
sim_time = 1000


#define populations
inp = sim.Population(input_pop,sim.SpikeSourcePoisson())
reg = sim.Population(adapitveA,sim)

#define connection
w = random.RandomDistribution('gamma',[10,0.001],random.NumpyRNG(seed=1234))
syn = sim.StaticSynapse(w,delay=0.05)

#Set

#connecting the network
connections = sim.Projection(inp,reg,sim.AllToAllConnector(),)
#Setup Simulation
sim.setup(timestep=0.01)

sim.run()
