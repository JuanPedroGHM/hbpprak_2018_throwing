#Function to throw the cylinder with use of the provided arm_control and hand_control scripts. 

import numpy as np

#initial values 
std_topology = [6,8,10,6]
std_weights = []
for index in range(len(std_topology)-1):
    std_weights.append(np.zeros(shape=(std_topology[index], std_topology[index+1])))

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
@nrp.MapVariable("topology", initial_value = std_topology )
@nrp.MapVariable("weights", initial_value = std_weights)


def throw_cylinder (t, arm_command, hand_command, 
                  topic_arm_1_pub, topic_arm_2_pub, topic_arm_3_pub, topic_arm_4_pub, topic_arm_5_pub, topic_arm_6_pub,
                  topic_arm_1_sub, topic_arm_2_sub, topic_arm_3_sub, topic_arm_4_sub, topic_arm_5_sub, topic_arm_6_sub,
                  topology,weights, arm_command, last_command_executed):
    
    pub_list = [topic_arm_1_pub, topic_arm_2_pub, topic_arm_3_pub, topic_arm_4_pub, topic_arm_5_pub, topic_arm_6_pub]
    sub_list = [topic_arm_1_sub, topic_arm_2_sub, topic_arm_3_sub, topic_arm_4_sub, topic_arm_5_sub, topic_arm_6_sub]
    
    
    import imp
    mod = imp.load_source('FeedForwardNetwork', '/home/bbpnrsoa/.opt/nrpStorage/hbpprak_2018_throwing/feedforward_network.py')
    
    network = mod.FeedForwardNetwork(topology.value)
    network.set_weights(weights.value)
    
    #get input to use for network
    network_inp = []
    for source in sub_list:
        elem = source.value
        if elem is not None:
            network_inp.append(source.value)
        else:
            return  
    
    clientLogger.info("The current input to the network is: " + str(network_inp))
    
    #use input to calculate output
    
    predictions = network.predict(network_inp)
    clientLogger.info("The network's output is : " + str(prediction))
    
    
    
    for index, prediction in enumerate(predictions):
        publist[index].send_message(std_msgs.msg.Float64(prediction))

