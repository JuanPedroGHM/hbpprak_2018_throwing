import numpy as np


@nrp.MapVariable("weights", initial_value = None, scope = nrp.GLOBAL)
@nrp.MapVariable("topology", initial_value = None, scope = nrp.GLOBAL)
@nrp.MapVariable("bias", initial_value = None, scope = nrp.GLOBAL)

def set_weights (t,weights,topology, bias):
    top = [6,10,8,6]
    in_wieghts = {}
    in_bias = {}
    for index in range(1,len(top)):
        in_wieghts['layer{}'.format(index)]=np.random.uniform(0,1,(top[index-1], top[index]))
        in_bias['layer{}'.format(index)]=np.random.uniform(0,1,(1,top[index]))
    weights.value = in_wieghts
    topology.value = top
    bias.value = in_bias
    