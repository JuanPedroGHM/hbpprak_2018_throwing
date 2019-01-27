import numpy as np

try:
    import _pickle as pickle
except ImportError:
    import cPickle as pickle


class FeedForwardNetwork(object):
    def __init__(self, layer_sizes, *args,**kwargs):
        self.weights = []
        self.bias = []
        
        for index in range(len(layer_sizes)-1):
            self.weights.append(np.random.uniform(-1,1,(layer_sizes[index], layer_sizes[index+1])))
            self.bias.append(np.random.uniform(-1,1, (1,layer_sizes[index+1])))

    def predict(self, inp,*args):
        out = np.expand_dims(inp.flatten(), 0)
        for weight, bias in zip(self.weights, self.bias):
            out = np.dot(out, weight) + bias
            out = np.arctan(out)
        return out[0]

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

    def get_bias(self):
        return self.bias

    def set_bias(self,bias):
       self.bias = bias

    def save(self, filename='weights.pkl'):
        with open(filename, 'wb') as fp:
            pickle.dump(self.weights, fp)

    def load(self, filename='weights.pkl'):
        with open(filename, 'rb') as fp:
            self.weights = pickle.load(fp)
