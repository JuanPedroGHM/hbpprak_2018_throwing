import pyNN.nest as sim
import numpy as np

try:
    import _pickle as pickle
except ImportError:
    import cPickle as pickle


class Spiking_feed_forward_network(object):
    def __init__(self,topology,min_fire_rate=0,max_fire_rate=100,simtime = 1000):

        self.topology = topology
        self.min_fire = min_fire_rate
        self.max_fire = max_fire_rate
        self.simtime = simtime
        self.weights = []

        for index in range(len(topology)-1):
            self.weights.append(np.ones(shape=(topology[index], topology[index+1])))

    def predict(self, inp, max_obs, min_obs):
        def get_fire_rate():
            return (inp-min_obs)/(max_obs-min_obs)*(self.max_fire-self.min_fire) + self.min_fire

        sim.setup(timestep=0.1,min_delay=0.1,threads=1)

        inp_pop = sim.Population(self.topology[0],sim.SpikeSourcePoisson())
        inp_pop.record('spikes')

        fire_rates = get_fire_rate()
        for index,neuron in enumerate(inp_pop):
            neuron.set_parameters(rate=fire_rates[index])

        nof_hidden_layers = len(self.topology)-2
        hidden_layers = []
        for i in range(nof_hidden_layers):
            cells = sim.Population(self.weights[i].shape[1], sim.IF_curr_exp())
            syn = sim.StaticSynapse(weight=self.weights[i], delay=0.5)
            if i==0:
                connections = sim.Projection(inp_pop, cells, sim.AllToAllConnector(allow_self_connections=False),syn)
            else:
                connections = sim.Projection(hidden_layers[i-1][0], cells, sim.AllToAllConnector(allow_self_connections=False),syn)
            hidden_layers.append((cells,syn,connections))
        out_pop = sim.Population(self.weights[nof_hidden_layers].shape[1],sim.IF_curr_exp())
        out_pop.record(('v','spikes'))
        syn = sim.StaticSynapse(weight=self.weights[nof_hidden_layers],delay=0.5)
        connections = sim.Projection(hidden_layers[-1][0],out_pop,sim.AllToAllConnector(allow_self_connections=False),syn)

        sim.run(self.simtime)

        inp_data = inp_pop.get_data()
        out_data = out_pop.get_data()

        return [len(spiketrain) for spiketrain in out_data.segments[0].spiketrains]

    def get_weights(self):
        return self.weights

    def set_weights(self,weights):
        self.weights = weights

    def set_fire_rate(self,**kwargs):
        min = kwargs.get("min")
        max = kwargs.get("max")

        if min is not None:
            self.min_fire = min
        if max is not None:
            self.max_fire = max
        if min is None and max is None:
            print("Could not assign Values for Firerate. Use **kwargs min or max")


    def get_fire_rate(self):
        return self.min_fire, self.max_fire

    def save(self, filename='weights.pkl'):
        with open(filename, 'wb') as fp:
            pickle.dump(self.weights, fp)

    def load(self, filename='weights.pkl'):
        with open(filename, 'rb') as fp:
            self.weights = pickle.load(fp)
