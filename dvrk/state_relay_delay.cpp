//
// Created by florian on 1/9/20.
//

#include <ros/ros.h>
#include <std_msgs/String.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/JointState.h>
#include <stdio.h>
#include <queue>

std::string lastStatePublished;
geometry_msgs::Pose lastPosePublishedPose;
geometry_msgs::PoseStamped lastPosePublishedPoseStamped;
sensor_msgs::JointState lastJawRecieved;

// Queues to buffer relayed messages
std::queue<std_msgs::String::ConstPtr> state_buffer;
std::queue<geometry_msgs::PoseStamped::ConstPtr> cart_buffer;
std::queue<sensor_msgs::JointState> jaw_buffer;

ros::Publisher state_pub;
ros::Publisher cartesian_pub;
ros::Publisher jaw_pub;

bool arePosesEqualPose(geometry_msgs::Pose pose1, geometry_msgs::Pose pose2){
    if (pose1.position.x != pose2.position.x){
        return false;
    }
    if (pose1.position.y != pose2.position.y){
        return false;
    }
    if (pose1.position.z != pose2.position.z){
        return false;
    }
    if (pose1.orientation.x != pose2.orientation.x){
        return false;
    }
    if (pose1.orientation.y != pose2.orientation.y){
        return false;
    }
    if (pose1.orientation.z != pose2.orientation.z){
        return false;
    }
    if (pose1.orientation.w != pose2.orientation.w){
        return false;
    }
    return true;
}

bool arePosesEqualPoseStamped(geometry_msgs::PoseStamped pose1, geometry_msgs::PoseStamped pose2){
    if (pose1.pose.position.x != pose2.pose.position.x){
        return false;
    }
    if (pose1.pose.position.y != pose2.pose.position.y){
        return false;
    }
    if (pose1.pose.position.z != pose2.pose.position.z){
        return false;
    }
    if (pose1.pose.orientation.x != pose2.pose.orientation.x){
        return false;
    }
    if (pose1.pose.orientation.y != pose2.pose.orientation.y){
        return false;
    }
    if (pose1.pose.orientation.z != pose2.pose.orientation.z){
        return false;
    }
    if (pose1.pose.orientation.w != pose2.pose.orientation.w){
        return false;
    }
    return true;
}

void stateCallback(const std_msgs::String::ConstPtr& msg){
    if(msg->data != lastStatePublished){
        state_pub.publish(msg);
        // state_buffer.push(msg);
        lastStatePublished = msg->data;
	std::cout << "New state: " << lastStatePublished << std::endl;
    }
}

void cartesianCallbackPose(const geometry_msgs::Pose::ConstPtr& msg){
    if(lastStatePublished == "READY" && !arePosesEqualPose(lastPosePublishedPose, *msg)){
        // cartesian_pub.publish(msg);
        // jaw_pub.publish(lastJawRecieved);
		
	geometry_msgs::PoseStamped::ConstPtr new_msg; 
	new_msg->header.stamp = ros::Time::now();
	new_msg->pose.position.x = msg->position.x; 
	new_msg->pose.position.y = msg->position.y;
	new_msg->pose.position.z = msg->position.z;
	new_msg->pose.orientation.x = msg->orientation.x; 
	new_msg->pose.orientation.y = msg->orientation.y;
	new_msg->pose.orientation.z = msg->orientation.z;
	new_msg->pose.orientation.w = msg->orientation.w;
        cart_buffer.push(new_msg);
        jaw_buffer.push(lastJawRecieved);

        lastPosePublishedPose = *msg;
    }
}

void cartesianCallbackPoseStamped(const geometry_msgs::PoseStamped::ConstPtr& msg){
    if(lastStatePublished == "READY" && !arePosesEqualPoseStamped(lastPosePublishedPoseStamped, *msg)){
        // cartesian_pub.publish(msg);
        // jaw_pub.publish(lastJawRecieved);

        cart_buffer.push(msg);
        jaw_buffer.push(lastJawRecieved);

        lastPosePublishedPoseStamped = *msg;
    }
}

void jawCallback(const sensor_msgs::JointState::ConstPtr& msg){
    lastJawRecieved = *msg;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "state_relay");
    ros::NodeHandle n;

    if(argc != 8){
        std::cerr << "Failed to launch state_relay, requires 6 arguments" << std::endl;
        std::cerr << "  -- Arg 1 = subscribed state topic name" << std::endl;
        std::cerr << "  -- Arg 2 = to publish state topic name" << std::endl;
        std::cerr << "  -- Arg 3 = subscribed cartesian topic name" << std::endl;
        std::cerr << "  -- Arg 4 = to publish cartesian topic name" << std::endl;
        std::cerr << "  -- Arg 5 = subscribed jaw topic name" << std::endl;
        std::cerr << "  -- Arg 6 = to publish jaw topic name" << std::endl;
	    std::cerr << "  -- Arg 7 = Pose or PoseStamped" << std::endl;
        return -1;
    }

    std::string state_subscribe_topic = argv[1];
    std::string state_publish_topic = argv[2];

    std::string cart_subscribe_topic = argv[3];
    std::string cart_publish_topic = argv[4];

    std::string jaw_subscribe_topic = argv[5];
    std::string jaw_publish_topic = argv[6];

    std::string pose_or_posestamped = argv[7]; 


    state_pub = n.advertise<std_msgs::String>   (state_publish_topic, 10);
    if (pose_or_posestamped == "Pose") {
	cartesian_pub = n.advertise<geometry_msgs::Pose>(cart_publish_topic,  10);
    } else if (pose_or_posestamped == "PoseStamped") {
	cartesian_pub = n.advertise<geometry_msgs::PoseStamped>(cart_publish_topic,  10);
    } else {
	std::cerr << "Arg 7 can only be Pose or PoseStamped!!!" << std::endl;
    } 
    jaw_pub = n.advertise<sensor_msgs::JointState>(jaw_publish_topic, 10);

    ros::Subscriber state_sub = n.subscribe(state_subscribe_topic, 10, stateCallback);
    ros::Subscriber cart_sub;
    if (pose_or_posestamped == "Pose") {
    	cart_sub  = n.subscribe(cart_subscribe_topic,  10, cartesianCallbackPose);
    } else if (pose_or_posestamped == "PoseStamped") {
	cart_sub  = n.subscribe(cart_subscribe_topic,  10, cartesianCallbackPoseStamped);
    } else {
	std::cerr << "Arg 7 can only be Pose or PoseStamped!!!" << std::endl;
    }
    ros::Subscriber jaw_sub = n.subscribe(jaw_subscribe_topic,   10, jawCallback);

    // ros::spin();

    // My new code goes here?
    ros::Rate loop_rate(100); // 10 Hz loop rate
    ros::Duration delay(1.0); // hardcode delay for now

    while (ros::ok()) {
        ros::spinOnce();
        ros::Time current_time = ros::Time::now();
	ros::Time front_timestamp(cart_buffer.front()->header.stamp.sec, cart_buffer.front()->header.stamp.nsec);

	// front_timestamp = cart_buffer.front()->header.stamp.sec + std::pow(cart_buffer.front()->header.stamp.nsec, -9);
        while (!cart_buffer.empty() && (current_time - front_timestamp) >= delay) {
            cartesian_pub.publish(cart_buffer.front());
            jaw_pub.publish(jaw_buffer.front());
            cart_buffer.pop();
            jaw_buffer.pop();
        }
        
        loop_rate.sleep();
    
    }
    
    return 0;
}
