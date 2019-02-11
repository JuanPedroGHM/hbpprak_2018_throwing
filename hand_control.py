# Only 8 joints are actuated, the rest are passive
# values taken from the urdf model at http://wiki.ros.org/schunk_svh_driver

@nrp.MapRobotPublisher("topic_index_proximal", Topic('/robot/hand_Index_Finger_Proximal/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("topic_index_distal", Topic('/robot/hand_Index_Finger_Distal/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("topic_middle_proximal", Topic('/robot/hand_Middle_Finger_Proximal/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("topic_middle_distal", Topic('/robot/hand_Middle_Finger_Distal/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("topic_ring_proximal", Topic('/robot/hand_Ring_Finger/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("topic_ring_distal", Topic('/robot/hand_j12/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("topic_pinky_proximal", Topic('/robot/hand_Pinky/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("topic_pinky_distal", Topic('/robot/hand_j13/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("topic_thumb_opposition", Topic('/robot/hand_Thumb_Opposition/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("topic_thumb_flexion", Topic('/robot/hand_j4/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher("topic_thumb_distal", Topic('/robot/hand_j3/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotSubscriber('command', Topic('/arm_robot/hand_commands', std_msgs.msg.String))
@nrp.MapVariable("last_command_executed", initial_value=None)
@nrp.Neuron2Robot()
def hand_control(t, command, last_command_executed,
                 topic_index_proximal, topic_index_distal,
                 topic_middle_proximal, topic_middle_distal,
                 topic_ring_proximal, topic_ring_distal,
                 topic_pinky_proximal, topic_pinky_distal,
                 topic_thumb_flexion, topic_thumb_distal,
                 topic_thumb_opposition):

    if command.value is None:
        return
    else:
        command_str = command.value.data

    if command_str == last_command_executed.value:
        return

    clientLogger.info("HAND received: {}".format(command_str))

    hand_topics = [
        topic_index_proximal,
        topic_index_distal,
        topic_middle_proximal,
        topic_middle_distal,
        topic_ring_proximal,
        topic_ring_distal,
        topic_pinky_proximal,
        topic_pinky_distal,
        topic_thumb_flexion,
        topic_thumb_distal]

    def do_grasp(strength):
        for topic in hand_topics:
            topic.send_message(std_msgs.msg.Float64(strength))

    topic_thumb_opposition.send_message(std_msgs.msg.Float64(1.2))

    last_command_executed.value = command_str

    if command_str == "GRASP":
        do_grasp(0.8)
    elif command_str == "RELEASE":
        do_grasp(0)

