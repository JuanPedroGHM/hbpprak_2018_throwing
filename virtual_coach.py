from evolution_strategy import EvolutionStrategy

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
        wdic["layer"+str(index)]=weights.tolist()
        index = index + 1
    return wdic

csv_name = 'cylinder_position.csv'
    
def save_position_csv(sim, datadir):
    with open(os.path.join(datadir, csv_name), 'wb') as f:
        cf = csv.writer(f)
        #################################################
        # Insert code here:
        # get the CSV data from the simulation
        #################################################
        csv_data = sim.get_csv_data(csv_name) #solution
        cf.writerows(csv_data)
    
# The function make_on_status() returns a on_status() function
# This is called a "closure": 
# it is here used to pass the sim and datadir objects to on_status()
def make_on_status(sim, datadir):
    def on_status(msg):
        print("Current simulation time: {}".format(msg['simulationTime']))
        if msg['simulationTime'] == 5.0 and sim.get_state() == 'started':
            #################################################
            # Insert code here:
            # 1) pause the simulation, 
            # 2) save the CSV file
            # 3) stop the simulation 
            #################################################
            sim.pause()  #solution
            save_position_csv(sim, datadir)
            sim.stop() #solution
            print("Trial terminated - saved CSV in {}".format(datadir))
            
    return on_status
        
def run_experiment(datadir, brain_params={'syn_weight': 1.0}):
    #################################################
    # Insert code here:
    # 1) launch the experiment
    # 2) add the status callback
    # 3) add the parametrized brain file
    # 4) add the extra CSV TF
    # 5) start the simulation
    #################################################
    brain_file = brain_template.format(**brain_params)
    
    sim = vc.launch_experiment('tutorial_baseball_solution_0') #solution
    sim.register_status_callback(make_on_status(sim, datadir)) #solution
    sim.add_transfer_function(record_ball_tf) #solution
    sim.edit_brain(brain_file) #solution
    sim.start()
    return sim
    


sim = vc.launch_experiment('hbpprak_2018_throwing')
sim.print_transfer_functions()


topology = [6,10,10,6]
weights = []

for index in range(len(topology)-1):
    weights.append(np.random.uniform(0,1,(topology[index], topology[index+1])))

#Start the evolutionary strategy

wdic = make_dict_from_weights(weights)

tf = 'import numpy as np\n@nrp.MapVariable("weights", initial_value = None, scope = nrp.GLOBAL)\n@nrp.MapVariable("topology", initial_value = None, scope = nrp.GLOBAL)\ndef set_weights (t,weights,topology):\n    weights.value = {}\n    topology.value = {}\n'.format(wdic,topology)

sim.edit_transfer_function('set_weights', tf)


tmp_folder = tempfile.mkdtemp()
sim=run_experiment(datadir=tmp_folder, )


sim.stop()
