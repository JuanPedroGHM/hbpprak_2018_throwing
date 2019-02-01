import numpy as np


@nrp.MapVariable("weights", initial_value = None, scope = nrp.GLOBAL)
@nrp.MapVariable("topology", initial_value = None, scope = nrp.GLOBAL)
@nrp.MapVariable("bias", initial_value = None, scope = nrp.GLOBAL)

def set_weights (t,weights,topology, bias):
    top = [6,10,8,6]
    in_wieghts = []
    in_bias = []
    for index in range(len(top)-1):
        in_wieghts.append(np.random.uniform(0,1,(top[index], top[index+1])))
        in_bias.append(np.random.uniform(0,1,(1,top[index+1])))
    weights.value = in_wieghts
    topology.value = top
    bias.value = in_bias
    