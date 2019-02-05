import numpy as np
from evolution_strategy import EvolutionStrategy
import tempfile
import os
import csv
import time
import pickle

def make_dict_from_weights(weights):
    index = 1
    wdic = {}
    for layer in weights:
        wdic["layer"+str(index)]=layer.tolist()
        index = index + 1
    return wdic

last_status = [None]
    
def on_status(msg):
    last_status[0] = msg


def wait_condition(timeout, description, condition):
    start = time.time()
    while time.time() < start + timeout:
	time.sleep(0.25)
	if condition(last_status[0]):
	    return
    raise TestCaseError(description)


def make_get_reward(sim, csv_name):

    def get_reward(index, weight, bias, topology):
        
        wdic = make_dict_from_weights(weights)
        wbias = make_dict_from_weights(bias)
        
        tf = 'import numpy as np\n@nrp.MapVariable("weights", initial_value = None, scope = nrp.GLOBAL)\n@nrp.MapVariable("topology", initial_value = None, scope = nrp.GLOBAL)\n@nrp.MapVariable("bias", initial_value = None, scope = nrp.GLOBAL)\ndef set_weights (t,weights,topology, bias):\n    top = [6,10,8,6]\n    in_wieghts = []\n    in_bias = []\n    weights.value = {}\n    topology.value = {}\n    bias.value = {}\n'.format(wdic, topology, wbias)
    
        sim.edit_transfer_function('set_weights',tf) 
        sim.start()
	
	wait_condition(100, 'Running simulation for 10 seconds', lambda x: x['simulationTime'] > 10.0)
	sim.pause()
            
        csv_data = np.array(sim.get_csv_data(csv_name))
        sim.reset('full')
	wait_condition(100, 'Waiting for full reset', lambda x: x['simulationTime'] == 0.0 and x['state'] == 'paused')
        return -float(csv_data[-1][2]) 

    return get_reward
    

if __name__ == '__main__':
    
    # Start simulation and launch experiment
    csv_name = "cylinder_position.csv"

    try:
        from hbp_nrp_virtual_coach.virtual_coach import VirtualCoach
        vc = VirtualCoach(environment='local', storage_username='nrpuser')
    except ImportError as e:
        print(e)
        print("You have to start this notebook with the command:\
              cle-virtual-coach jupyter notebook")
        raise e

    sim = vc.launch_experiment('hbpprak_2018_throwing')

    sim.register_status_callback(on_status) #solution

    # Network params and init
    
    topology = [6,100,20,6]

    weights = []
    bias = []
    rewards = []

    for index in range(len(topology)-1):
        weights.append(np.random.uniform(-1,1,(topology[index], topology[index+1])))
        bias.append(np.random.uniform(-1,1,(1,topology[index+1])))


    #Start the evolutionary strategy
    #Evo Params

    n_threads = 1
    pop_size = 2
    learning_rate = 0.03
    decay = 0.999
    sigma = 0.2
    iterations = 2

    es = EvolutionStrategy(topology, weights, bias, make_get_reward(sim, csv_name), pop_size, sigma, learning_rate, decay, n_threads)

    average_rewards = es.run(iterations, 1)

    final_weights = es.get_weights()
    final_bias = es.bias

    optimal_params = {'weights':final_weights,'bias':final_bias}
    with open("optimal_params.pickle","wb") as f:
        pickle.dump(optimal_params,f)

    np.savetxt("rewards.txt",np.array(average_rewards))

