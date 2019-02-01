import numpy as np
from evolution_strategy import EvolutionStrategy
import tempfile
import os

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

csv_name = 'cylinder_position.csv'
    
def save_position_csv(sim, datadir):
    with open(os.path.join(datadir, csv_name), 'wb') as f:
        cf = csv.writer(f)
        csv_data = sim.get_csv_data(csv_name) #solution
        cf.writerows(csv_data)
    
def make_on_status(sim, datadir):
    def on_status(msg):
        print("Current simulation time: {}".format(msg['simulationTime']))
        if msg['simulationTime'] == 10.0 and sim.get_state() == 'started':
            sim.pause()  #solution
            save_position_csv(sim, datadir)
            sim.stop() #solution
            print("Trial terminated - saved CSV in {}".format(datadir))
            
    return on_status
        
def run_experiment(datadir, weights, bias, topology):
    
    wdic = make_dict_from_weights(weights)
    wbias = make_dict_from_weights(bias)
    
    tf = 'import numpy as np\n@nrp.MapVariable("weights", initial_value = None, scope = nrp.GLOBAL)\n@nrp.MapVariable("topology", initial_value = None, scope = nrp.GLOBAL)\n@nrp.MapVariable("bias", initial_value = None, scope = nrp.GLOBAL)\ndef set_weights (t,weights,topology, bias):\n    top = [6,10,8,6]\n    in_wieghts = []\n    in_bias = []\n    weights.value = {}\n    topology.value = {}\n    bias.value = {}\n'.format(wdic, topology, wbias)
    
   
    sim = vc.launch_experiment('hbpprak_2018_throwing')
    sim.register_status_callback(make_on_status(sim, datadir)) #solution
     
    sim.edit_transfer_function('set_weights',tf) #solution
    buf = sim.get_transfer_function('throw_cylinder')
    sim.edit_transfer_function('throw_cylinder',buf) #solution
    sim.start()
    return sim
    

topology = [6,10,10,6]
weights = []
bias = []

for index in range(len(topology)-1):
    weights.append(np.random.uniform(0,1,(topology[index], topology[index+1])))
    bias.append(np.random.uniform(0,1,(1,topology[index+1])))


#Start the evolutionary strategy


tmp_folder = tempfile.mkdtemp()
sim=run_experiment(tmp_folder, weights, bias, topology)
#sim.stop()#
csv_file = os.path.join(tmp_folder, csv_name)
print(csv_file)
