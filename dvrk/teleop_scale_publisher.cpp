#include <ros/ros.h>
#include <std_msgs/Float32.h>
#include <iostream>

int main(int argc, char** argv)
{
    // Initialize the ROS system
    ros::init(argc, argv, "teleop_scale_publisher");
    ros::NodeHandle nh;

    // Create a publisher object
    ros::Publisher pub = nh.advertise<std_msgs::Float32>("/teleop/set_scale", 1000);

    // Loop rate (not necessary for this example as we use blocking I/O)
    ros::Rate rate(10);

    while (ros::ok())
    {
        // Take user input
        std::cout << "Enter a float value for set_scale: ";
        float input_value;
        std::cin >> input_value;

        // Check for input failure
        if(std::cin.fail())
        {
            std::cin.clear(); // clear the error flag
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // discard bad input
            std::cerr << "Invalid input. Please enter a float value." << std::endl;
            continue;
        }

        // Create a Float32 message
        std_msgs::Float32 msg;
        msg.data = input_value;

        // Publish the message
        pub.publish(msg);

        // Spin once to allow callbacks (not used in this simple script)
        ros::spinOnce();

        // Sleep for a short time to avoid spamming (optional)
        rate.sleep();
    }

    return 0;
}
