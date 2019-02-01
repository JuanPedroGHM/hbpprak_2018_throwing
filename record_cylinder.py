import numpy as np
import sensor_msgs.msg

@nrp.MapCSVRecorder("cylinder_recorder", filename="cylinder_position.csv",
                    headers=["Time", "px", "py", "pz"])
@nrp.Robot2Neuron()
def record_cylinder (t,cylinder_recorder):
    from rospy import ServiceProxy
    from gazebo_msgs.srv import GetModelState

    model_name = 'cylinder'
    state_proxy = ServiceProxy('/gazebo/get_model_state',
                                    GetModelState, persistent=False)
    cylinder = state_proxy(model_name, "world")

    if cylinder.success:
        current_position = cylinder.pose.position
        cylinder_recorder.record_entry(t,
                                   current_position.x, 
                                   current_position.y, 
                                   current_position.z)
    
