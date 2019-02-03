import numpy as np
from evolution_strategy import EvolutionStrategy
import tempfile
import os
import csv
import time
import pickle

try:
    from hbp_nrp_virtual_coach.virtual_coach import VirtualCoach
    vc = VirtualCoach(environment='local', storage_username='nrpuser')
except ImportError as e:
    print(e)
    print("You have to start this notebook with the command:\
          cle-virtual-coach jupyter notebook")
    raise e

def make_dict_from_weights(weights):
    index = 1
    wdic = {}
    for layer in weights:
        wdic["layer"+str(index)]=layer.tolist()
        index = index + 1
    return wdic
    
def make_on_status(sim, datadir, index):
    def on_status(msg):
        print("Current simulation time: {}".format(msg['simulationTime']))
        if msg['simulationTime'] == 10.0 and sim.get_state() == 'started':
            sim.pause()  #solution

            with open(os.path.join(datadir, 'cylinder_position_{}.csv'.format(index)), 'wb') as f:
                cf = csv.writer(f)
                csv_data = sim.get_csv_data(csv_name) #solution
                cf.writerows(csv_data)
                
            sim.stop() #solution
            print("Trial terminated - saved CSV in {}".format(datadir))
        
    return on_status

def make_get_reward(tmp_folder):
    def get_reward(index, weight, bias, topology):
        
        
        sim=run_experiment(tmp_folder, index, weights, bias, topology)
        while sim.get_state() != 'stopped':
            pass
        try:
            filePath = os.path.join(tmp_folder, 'cylinder_position_{}.csv'.format(index))
            data = np.genfromtxt(filePath, delimiter = ",")
            end_point = data[-1][2]
        except AttributeError:
            print("Could not read reward from csv file {}".format(filePath))
            return None
        
        time.sleep(10)    
        return -end_point 
    
    return get_reward

def run_experiment(datadir, index, weights, bias, topology):
    
    wdic = make_dict_from_weights(weights)
    wbias = make_dict_from_weights(bias)
    
    tf = 'import numpy as np\n@nrp.MapVariable("weights", initial_value = None, scope = nrp.GLOBAL)\n@nrp.MapVariable("topology", initial_value = None, scope = nrp.GLOBAL)\n@nrp.MapVariable("bias", initial_value = None, scope = nrp.GLOBAL)\ndef set_weights (t,weights,topology, bias):\n    top = [6,10,8,6]\n    in_wieghts = []\n    in_bias = []\n    weights.value = {}\n    topology.value = {}\n    bias.value = {}\n'.format(wdic, topology, wbias)
    
   
    sim = vc.launch_experiment('hbpprak_2018_throwing')
    sim.register_status_callback(make_on_status(sim, datadir, index)) #solution
     
    sim.edit_transfer_function('set_weights',tf) #solution
    buf = sim.get_transfer_function('throw_cylinder')
    sim.edit_transfer_function('throw_cylinder',buf) #solution
    sim.start()

    return sim
    

topology = [6,100,20,6]
weights = []
bias = []
rewards = []

for index in range(len(topology)-1):
    weights.append(np.random.uniform(0,1,(topology[index], topology[index+1])))
    bias.append(np.random.uniform(0,1,(1,topology[index+1])))


#Start the evolutionary strategy
#Evo Params

csv_name = "cylinder_position.csv"
n_threads = 1
pop_size = 2
learning_rate = 0.03
decay = 0.999
sigma = 0.2
iterations = 2

path = '/home/bbpnrsoa/.opt/nrpStorage/hbpprak_2018_throwing/results'

print("The temporary folder for this experiment is {}".format(path))

es = EvolutionStrategy(topology, weights, bias, make_get_reward(path), pop_size, sigma, learning_rate, decay, n_threads)

average_rewards = es.run(iterations, 1)

final_weights = es.get_weights()
final_bias = es.bias

optimal_params = {'weights':final_weights,'bias':final_bias}
with open("optimal_params.pickle","wb") as f:
    pickle.dump(optimal_params,f)

np.savetxt("rewards.txt",np.array(average_rewards))

