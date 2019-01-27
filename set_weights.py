import numpy as np


@nrp.MapVariable("weights", initial_value = None, scope = nrp.GLOBAL)
@nrp.MapVariable("topology", initial_value = None, scope = nrp.GLOBAL)

def set_weights (t,weights,topology):
    top = [6,10,8,6]
    in_wieghts = []
    for index in range(len(top)-1):
        in_wieghts.append(np.random.uniform(0,1,(top[index], top[index+1])))
    weights.value = in_wieghts
    topology.value = top
    