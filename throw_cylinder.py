# Imported Python Transfer Function
#Function to throw the cylinder with use of the provided arm_control and hand_control scripts. 
import numpy as np
#initial values 
std_topology = None
std_weights = None
std_bias = None
#Publish Topics
@nrp.MapRobotPublisher("topic_arm_1_pub", Topic('/robot/arm_1_joint/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("topic_arm_2_pub", Topic('/robot/arm_2_joint/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("topic_arm_3_pub", Topic('/robot/arm_3_joint/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("topic_arm_4_pub", Topic('/robot/arm_4_joint/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("topic_arm_5_pub", Topic('/robot/arm_5_joint/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("topic_arm_6_pub", Topic('/robot/arm_6_joint/cmd_pos', std_msgs.msg.Float64))
#Verwendung der bereitgestellten Funktionen
@nrp.MapRobotPublisher("arm_command", Topic('/arm_robot/arm_commands', std_msgs.msg.String))
@nrp.MapRobotPublisher("hand_command", Topic('/arm_robot/hand_commands', std_msgs.msg.String))
#Subscripe 
@nrp.MapRobotSubscriber("topic_arm_1_sub", Topic('/robot/arm_1_joint/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotSubscriber("topic_arm_2_sub", Topic('/robot/arm_2_joint/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotSubscriber("topic_arm_3_sub", Topic('/robot/arm_3_joint/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotSubscriber("topic_arm_4_sub", Topic('/robot/arm_4_joint/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotSubscriber("topic_arm_5_sub", Topic('/robot/arm_5_joint/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotSubscriber("topic_arm_6_sub", Topic('/robot/arm_6_joint/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotSubscriber('arm_command', Topic('/arm_robot/arm_commands', std_msgs.msg.String))
@nrp.MapVariable("last_command_executed", initial_value=None)
#Global Varibales
@nrp.MapVariable("weights", initial_value = std_weights, scope = nrp.GLOBAL)
@nrp.MapVariable("topology", initial_value = std_topology, scope = nrp.GLOBAL)
@nrp.MapVariable("bias", initial_value = std_topology, scope = nrp.GLOBAL)

@nrp.Robot2Neuron(throttling_rate=15.0)
def throw_cylinder (t, arm_command, hand_command, 
                  topic_arm_1_pub, topic_arm_2_pub, topic_arm_3_pub, topic_arm_4_pub, topic_arm_5_pub, topic_arm_6_pub,
                  topic_arm_1_sub, topic_arm_2_sub, topic_arm_3_sub, topic_arm_4_sub, topic_arm_5_sub, topic_arm_6_sub,
                  weights, topology, bias, arm_command, last_command_executed):
    pub_list = [topic_arm_1_pub, topic_arm_2_pub, topic_arm_3_pub, topic_arm_4_pub, topic_arm_5_pub, topic_arm_6_pub]
    sub_list = [topic_arm_1_sub, topic_arm_2_sub, topic_arm_3_sub, topic_arm_4_sub, topic_arm_5_sub, topic_arm_6_sub]
    if topology.value is None or weights.value is None:
        return
    #clientLogger.info("The topology is:" + str(topology.value))
    import imp
    mod = imp.load_source('FeedForwardNetwork', '/home/nrpuser/.opt/nrpStorage/template_manipulation_0/feedforward_network.py')
    network = mod.FeedForwardNetwork(topology.value)
    #Create the weights array from weights.value
    weight_arr = []
    #Use this method, if the experiment gets run by the virtual coach
    #for key in weights.key():
    #    weight_arr.append(weights[key])
    #network.set_weights(np.array(weight_arr))
    weight_inp = []
    bias_inp = []
    layer_size = len(topology.value) - 1 
    for i in range(1,layer_size+1):
        weight_inp.append(np.array(weights.value['layer{}'.format(i)]))
        bias_inp.append(np.array(bias.value['layer{}'.format(i)]))
    network.set_weights(weight_inp)
    network.set_bias(bias_inp)
    #get input to use for network
    network_inp = []
    for source in sub_list:
        elem = source.value
        #clientLogger.info(elem)
        if elem is not None:
            network_inp.append(source.value.data)
        else:
            return
    clientLogger.info("The current input to the network is: {}".format(network_inp))
    #clientLogger.info(weights.value)
    #use input to calculate output
    predictions = network.predict(np.array(network_inp), clientLogger)
    clientLogger.info("The network's output is : " + str(predictions))
    #send the output to the joints of the robot
    for index, prediction in enumerate(predictions):
        pub_list[index].send_message(std_msgs.msg.Float64(prediction * 1.2))
