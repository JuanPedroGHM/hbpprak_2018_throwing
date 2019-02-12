import numpy as np
from evolution_strategy import EvolutionStrategy
import tempfile
import os
import csv
import time
import pickle
import argparse

class TestCaseError(Exception):
    pass

def make_dict_from_weights(weights):
    index = 1
    wdic = {}
    for layer in weights:
        wdic["layer"+str(index)]=layer.tolist()
        index = index + 1
    return wdic

    



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('file')

    args = parser.parse_args()
    filename = args.file
    
    try:
        from hbp_nrp_virtual_coach.virtual_coach import VirtualCoach
        vc = VirtualCoach(environment='local', storage_username='nrpuser')
    except ImportError as e:
        print(e)
        print("You have to start this notebook with the command:\
              cle-virtual-coach jupyter notebook")
        raise e

    sim = vc.launch_experiment('hbpprak_2018_throwing')

    topology = [6,100,20,6]

    weights = []
    bias = []
    rewards = []

    #with open("tmp_weights_grasp.pickle","rb") as f:
    with open(filename,"rb") as f:
        a = pickle.load(f)
        weights = a['weights']
        bias = a['bias']

    #Start the evolutionary strategy

    #Evo Params
    
    wdic = make_dict_from_weights(weights)
    wbias = make_dict_from_weights(bias)
    
    tf = 'import numpy as np\n@nrp.MapVariable("weights", initial_value = None, scope = nrp.GLOBAL)\n@nrp.MapVariable("topology", initial_value = None, scope = nrp.GLOBAL)\n@nrp.MapVariable("bias", initial_value = None, scope = nrp.GLOBAL)\ndef set_weights (t,weights,topology, bias):\n    top = [6,10,8,6]\n    in_wieghts = []\n    in_bias = []\n    weights.value = {}\n    topology.value = {}\n    bias.value = {}\n'.format(wdic, topology, wbias)

    sim.add_transfer_function(tf)

