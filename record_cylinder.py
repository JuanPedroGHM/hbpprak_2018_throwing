# Imported Python Transfer Function
import numpy as np
import sensor_msgs.msg
@nrp.MapCSVRecorder("cylinder_recorder", filename="cylinder_position.csv",
                    headers=["Time", "py", "dist"])
@nrp.Robot2Neuron(throttling_rate=5)
def record_cylinder (t,cylinder_recorder):
    from rospy import ServiceProxy
    from gazebo_msgs.srv import GetModelState, GetLinkState
    from gazebo_msgs.msg import LinkState
    model_name = 'cylinder'
    state_proxy = ServiceProxy('/gazebo/get_model_state', GetModelState, persistent=False)
    position_proxy = ServiceProxy('/gazebo/get_link_state', GetLinkState, persistent=False)
    cylinder = state_proxy(model_name, "world")
    
    hand = position_proxy('robot::hand_f3_link', 'world')
 
    if cylinder.success and hand.success:
        cylinder_position = np.array([cylinder.pose.position.x, cylinder.pose.position.y, cylinder.pose.position.z])
	hand_position = np.array([hand.link_state.pose.position.x, hand.link_state.pose.position.y,hand.link_state.pose.position.z])
	
	distance = np.linalg.norm(cylinder_position - hand_position)
	
        cylinder_recorder.record_entry(t,
                                   cylinder_position[1], 
                                   distance)
