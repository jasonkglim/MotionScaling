//
// Created by florian on 1/9/20.
//

#include <ros/ros.h>
#include <std_msgs/String.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/JointState.h>
#include <stdio.h>

#include <boost/circular_buffer.hpp>

int buffer_size = 500;

std::string lastStatePublished;

geometry_msgs::Pose lastPosePublished;
sensor_msgs::JointState lastJawPublished;

boost::circular_buffer<geometry_msgs::Pose> pose_buffer = boost::circular_buffer<geometry_msgs::Pose>(buffer_size);
boost::circular_buffer<geometry_msgs::PoseStamped> pose_buffer_stamped = boost::circular_buffer<geometry_msgs::PoseStamped>(buffer_size);

boost::circular_buffer<ros::Time> ts_pose_buffer           = boost::circular_buffer<ros::Time>(buffer_size);

boost::circular_buffer<sensor_msgs::JointState> jaw_buffer = boost::circular_buffer<sensor_msgs::JointState>(buffer_size);
boost::circular_buffer<ros::Time> ts_jaw_buffer            = boost::circular_buffer<ros::Time>(buffer_size);


ros::Publisher state_pub;
ros::Publisher cartesian_pub;
ros::Subscriber cart_sub;
ros::Publisher jaw_pub;

//Only checks position
bool areJointStatesEqual(sensor_msgs::JointState j1, sensor_msgs::JointState j2){

    if(j1.position.size() != j2.position.size()){
        return false;
    }

    for(int i = 0; i <= j1.position.size();i++){
        if( j1.position[i] != j2.position[i]){
            return false;
        }
    }
    return true;
}

bool arePosesEqual(geometry_msgs::Pose pose1, geometry_msgs::Pose pose2){
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

void stateCallback(const std_msgs::String::ConstPtr& msg){

    if(msg->data != lastStatePublished){
        state_pub.publish(msg);
        lastStatePublished = msg->data;
    }
}

void cartesianCallback(const geometry_msgs::Pose::ConstPtr& msg){
    pose_buffer.push_back(*msg);
    ts_pose_buffer.push_back(ros::Time::now());
}

void cartesianStampedCallback(const geometry_msgs::PoseStamped::ConstPtr& msg){
    pose_buffer_stamped.push_back(*msg);
    ts_pose_buffer.push_back(ros::Time::now());
}


void jawCallback(const sensor_msgs::JointState::ConstPtr& msg){
    jaw_buffer.push_back(*msg);
    ts_jaw_buffer.push_back(ros::Time::now());
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "state_relay");
    ros::NodeHandle n;

    if(argc != 10){
        std::cerr << "Failed to launch state_relay, requires 8 arguments" << std::endl;
        std::cerr << "  -- Arg 1 = subscribed state topic name" << std::endl;
        std::cerr << "  -- Arg 2 = to publish state topic name" << std::endl;
        std::cerr << "  -- Arg 3 = subscribed cartesian topic name" << std::endl;
        std::cerr << "  -- Arg 4 = to publish cartesian topic name" << std::endl;
        std::cerr << "  -- Arg 5 = subscribed jaw topic name" << std::endl;
        std::cerr << "  -- Arg 6 = to publish jaw topic name" << std::endl;
        std::cerr << "  -- Arg 7 = additional delay to relay for cartesian and jaw topics" << std::endl;
        std::cerr << "  -- Arg 8 = to use state READY or not (1 or 0)" << std::endl;
        std::cerr << "  -- Arg 9 = to use pose stamped or not (1 or 0)" << std::endl;
        return -1;
    }

    std::string state_subscribe_topic = argv[1];
    std::string state_publish_topic   = argv[2];

    std::string cart_subscribe_topic = argv[3];
    std::string cart_publish_topic   = argv[4];

    std::string jaw_subscribe_topic = argv[5];
    std::string jaw_publish_topic   = argv[6];
    float delay  = boost::lexical_cast<float>(argv[7]);
    // Seems like if useReady is 1, we don't need to check that lastStatePublished is READY to publish pose
    // If useReady is 0, we should check that lastStatePublished is READY before publishing pose
    int useReady = boost::lexical_cast<int>(argv[8]);
    int usePoseStamped = boost::lexical_cast<int>(argv[9]);



    state_pub     = n.advertise<std_msgs::String>   (state_publish_topic, 10);
    jaw_pub       = n.advertise<sensor_msgs::JointState>(jaw_publish_topic, 10);


    ros::Subscriber state_sub = n.subscribe(state_subscribe_topic, 10, stateCallback);

    if(usePoseStamped != 0){
        cart_sub = n.subscribe(cart_subscribe_topic,  10, cartesianStampedCallback);
        cartesian_pub = n.advertise<geometry_msgs::PoseStamped>(cart_publish_topic,  10);
    }
    else{
        cart_sub = n.subscribe(cart_subscribe_topic,  10, cartesianCallback);
        cartesian_pub = n.advertise<geometry_msgs::Pose>(cart_publish_topic,  10);
    }

    ros::Subscriber jaw_sub   = n.subscribe(jaw_subscribe_topic,   10, jawCallback);

    ros::Rate loop_rate(500);
    geometry_msgs::Pose curPose;

    while (ros::ok()) {
        ros::spinOnce();
        loop_rate.sleep();

        // check if anything is in pose buffer
        if (!ts_pose_buffer.empty()){
            while ((ros::Time::now() - ts_pose_buffer.front()).toSec() > delay) {

                // set curPose to either front of pose or poseStamped buffer
                if(usePoseStamped != 0){
                    curPose = pose_buffer_stamped.front().pose;
                }
                else{
                    curPose = pose_buffer.front();
                }

                // If we want to useReady, useReady == 1
                if(useReady != 0){
                    // just check that last published pose is different from front of our buffer
                    if(!arePosesEqual(lastPosePublished, curPose)){
                        // publish either front of pose or posestamped, update lastPosePublished
                        if(usePoseStamped != 0){
                            cartesian_pub.publish(pose_buffer_stamped.front());
                        }
                        else{
                            cartesian_pub.publish(curPose);
                        }
                        lastPosePublished = curPose;
                    }
                }
                // if useReady is 0, we check also check that lastStatePublished is READY, everything else stays the same
                else{
                    if (lastStatePublished == "READY" && !arePosesEqual(lastPosePublished, curPose)){
                        if(usePoseStamped != 0){
                            cartesian_pub.publish(pose_buffer_stamped.front());
                        }
                        else{
                            cartesian_pub.publish(curPose);
                        }
                        lastPosePublished = curPose;
                    }
                }

                // Pop proper buffer
                if(usePoseStamped != 0){
                    pose_buffer_stamped.pop_front();
                }
                else{
                    pose_buffer.pop_front();
                }

                ts_pose_buffer.pop_front();

                // Continue while loop checking front of buffer unless its empty
                if(ts_pose_buffer.empty()){
                    break;
                }
            }
        }

        // Repeat for jaw buffer
        if(!ts_jaw_buffer.empty()) {
            while ((ros::Time::now() - ts_jaw_buffer.front()).toSec() > delay) {

                if(useReady != 0){
                    if(!arePosesEqual(lastPosePublished, pose_buffer.front())){
                        jaw_pub.publish(jaw_buffer.front());
                        lastJawPublished = jaw_buffer.front();
                    }
                }
                else{
                    if (lastStatePublished == "READY" && !arePosesEqual(lastPosePublished, pose_buffer.front())){
                        jaw_pub.publish(jaw_buffer.front());
                        lastJawPublished = jaw_buffer.front();
                    }
                }

                jaw_buffer.pop_front();
                ts_jaw_buffer.pop_front();

                if(ts_jaw_buffer.empty()){
                    break;
                }

            }
        }
    }


    return 0;
}
