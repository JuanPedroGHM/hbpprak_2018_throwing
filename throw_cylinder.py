# Imported Python Transfer Function
#Function to throw the cylinder with use of the provided arm_control and hand_control scripts. 
import numpy as np
#initial values 
std_topology = None
std_weights = None
std_bias = None

active = False
#Publish Topics
@nrp.MapRobotPublisher("topic_arm_1_pub", Topic('/robot/arm_1_joint/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("topic_arm_2_pub", Topic('/robot/arm_2_joint/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("topic_arm_3_pub", Topic('/robot/arm_3_joint/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("topic_arm_4_pub", Topic('/robot/arm_4_joint/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("topic_arm_5_pub", Topic('/robot/arm_5_joint/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("topic_arm_6_pub", Topic('/robot/arm_6_joint/cmd_pos', std_msgs.msg.Float64))
#Subscripe 
@nrp.MapRobotSubscriber("topic_arm_1_sub", Topic('/robot/arm_1_joint/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotSubscriber("topic_arm_2_sub", Topic('/robot/arm_2_joint/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotSubscriber("topic_arm_3_sub", Topic('/robot/arm_3_joint/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotSubscriber("topic_arm_4_sub", Topic('/robot/arm_4_joint/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotSubscriber("topic_arm_5_sub", Topic('/robot/arm_5_joint/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotSubscriber("topic_arm_6_sub", Topic('/robot/arm_6_joint/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotSubscriber("topic_Index_Proximal", Topic('/robot/hand_Index_Finger_Proximal/cmd_pos', std_msgs.msg.Float64))
#Activate networkt topic
@nrp.MapRobotSubscriber("topic_activate_network", Topic('/network/activate', std_msgs.msg.Bool))
@nrp.MapRobotPublisher("hand_command", Topic('/arm_robot/hand_commands', std_msgs.msg.String))
#Global Varibales
@nrp.MapVariable("weights", initial_value = std_weights, scope = nrp.GLOBAL)
@nrp.MapVariable("topology", initial_value = std_topology, scope = nrp.GLOBAL)
@nrp.MapVariable("bias", initial_value = std_topology, scope = nrp.GLOBAL)
@nrp.MapVariable("active", initial_value = False, scope = nrp.GLOBAL)

@nrp.Robot2Neuron(throttling_rate=5)
def throw_cylinder (t, topic_activate_network, 
                  topic_arm_1_pub, topic_arm_2_pub, topic_arm_3_pub, topic_arm_4_pub, topic_arm_5_pub, topic_arm_6_pub,
                  topic_arm_1_sub, topic_arm_2_sub, topic_arm_3_sub, topic_arm_4_sub, topic_arm_5_sub, topic_arm_6_sub, topic_Index_Proximal,
                  weights, topology, bias, active,
                   hand_command):
    pub_list = [topic_arm_1_pub, topic_arm_2_pub, topic_arm_3_pub, topic_arm_4_pub, topic_arm_5_pub, topic_arm_6_pub]
    sub_list = [topic_arm_1_sub, topic_arm_2_sub, topic_arm_3_sub, topic_arm_4_sub, topic_arm_5_sub, topic_arm_6_sub, topic_Index_Proximal]
    if topology.value is None or weights.value is None:
        return

    active = topic_activate_network.value.data if topic_activate_network.value != None else active
    if not active:
        return
    #clientLogger.info("The topology is:" + str(topology.value))
    import imp
    mod = imp.load_source('FeedForwardNetwork', '/home/bbpnrsoa/.opt/nrpStorage/hbpprak_2018_throwing/feedforward_network.py')
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
    network_inp[-1] = network_inp[-1] * 2
    predictions = network.predict(np.array(network_inp), clientLogger)
    clientLogger.info("The network's output is : " + str(predictions))
    #send the output to the joints of the robot
    for index, prediction in enumerate(predictions):
        if index == (len(predictions) - 1):
            hand_command.send_message(std_msgs.msg.String("GRASP_{}".format((prediction/2) + 1)))
        else:
            pub_list[index].send_message(std_msgs.msg.Float64(prediction * 1.5))
