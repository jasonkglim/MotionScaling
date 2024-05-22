#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState

lastStatePublished = ""
lastPosePublishedPose = Pose()
lastPosePublishedPoseStamped = PoseStamped()
lastJawReceived = JointState()

state_pub = None
cartesian_pub = None
jaw_pub = None

def are_poses_equal_pose(pose1, pose2):
    return (pose1.position.x == pose2.position.x and
            pose1.position.y == pose2.position.y and
            pose1.position.z == pose2.position.z and
            pose1.orientation.x == pose2.orientation.x and
            pose1.orientation.y == pose2.orientation.y and
            pose1.orientation.z == pose2.orientation.z and
            pose1.orientation.w == pose2.orientation.w)

def are_poses_equal_pose_stamped(pose1, pose2):
    return (pose1.pose.position.x == pose2.pose.position.x and
            pose1.pose.position.y == pose2.pose.position.y and
            pose1.pose.position.z == pose2.pose.position.z and
            pose1.pose.orientation.x == pose2.pose.orientation.x and
            pose1.pose.orientation.y == pose2.pose.orientation.y and
            pose1.pose.orientation.z == pose2.pose.orientation.z and
            pose1.pose.orientation.w == pose2.pose.orientation.w)

def state_callback(msg):
    global lastStatePublished
    if msg.data != lastStatePublished:
        state_pub.publish(msg)
        lastStatePublished = msg.data
        rospy.loginfo(f"New state: {lastStatePublished}")

def cartesian_callback_pose(msg):
    global lastPosePublishedPose
    if lastStatePublished == "READY" and not are_poses_equal_pose(lastPosePublishedPose, msg):
        cartesian_pub.publish(msg)
        jaw_pub.publish(lastJawReceived)
        lastPosePublishedPose = msg

def cartesian_callback_pose_stamped(msg):
    global lastPosePublishedPoseStamped
    if lastStatePublished == "READY" and not are_poses_equal_pose_stamped(lastPosePublishedPoseStamped, msg):
        cartesian_pub.publish(msg)
        jaw_pub.publish(lastJawReceived)
        lastPosePublishedPoseStamped = msg

def jaw_callback(msg):
    global lastJawReceived
    lastJawReceived = msg

def main():
    global state_pub, cartesian_pub, jaw_pub

    rospy.init_node('state_relay', anonymous=True)

    if len(rospy.myargv()) != 8:
        rospy.logerr("Failed to launch state_relay, requires 6 arguments")
        rospy.logerr("  -- Arg 1 = subscribed state topic name")
        rospy.logerr("  -- Arg 2 = to publish state topic name")
        rospy.logerr("  -- Arg 3 = subscribed cartesian topic name")
        rospy.logerr("  -- Arg 4 = to publish cartesian topic name")
        rospy.logerr("  -- Arg 5 = subscribed jaw topic name")
        rospy.logerr("  -- Arg 6 = to publish jaw topic name")
        rospy.logerr("  -- Arg 7 = Pose or PoseStamped")
        return -1

    state_subscribe_topic = rospy.myargv()[1] # e.g. PSM1
    state_publish_topic = rospy.myargv()[2] # e.g. NetPSM1

    cart_subscribe_topic = rospy.myargv()[3]
    cart_publish_topic = rospy.myargv()[4]

    jaw_subscribe_topic = rospy.myargv()[5]
    jaw_publish_topic = rospy.myargv()[6]

    pose_or_posestamped = rospy.myargv()[7]

    state_pub = rospy.Publisher(state_publish_topic, String, queue_size=10)
    jaw_pub = rospy.Publisher(jaw_publish_topic, JointState, queue_size=10)

    if pose_or_posestamped == "Pose":
        cartesian_pub = rospy.Publisher(cart_publish_topic, Pose, queue_size=10)
    elif pose_or_posestamped == "PoseStamped":
        cartesian_pub = rospy.Publisher(cart_publish_topic, PoseStamped, queue_size=10)
    else:
        rospy.logerr("Arg 7 can only be Pose or PoseStamped!!!")
        return -1

    rospy.Subscriber(state_subscribe_topic, String, state_callback)
    
    if pose_or_posestamped == "Pose":
        rospy.Subscriber(cart_subscribe_topic, Pose, cartesian_callback_pose)
    elif pose_or_posestamped == "PoseStamped":
        rospy.Subscriber(cart_subscribe_topic, PoseStamped, cartesian_callback_pose_stamped)
    else:
        rospy.logerr("Arg 7 can only be Pose or PoseStamped!!!")
        return -1

    rospy.Subscriber(jaw_subscribe_topic, JointState, jaw_callback)

    rospy.spin()

if __name__ == '__main__':
    main()
