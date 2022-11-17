#include <string>
#include <ros/ros.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <trajectory_msgs/JointTrajectoryPoint.h>
// #include <control_msgs/JointTrajectoryControllerState.h>
#include <sensor_msgs/JointState.h>
#include <visualization_msgs/Marker.h>
#include <nav_msgs/Odometry.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>

#include <eigenutils/eigen_file_io.hpp>

#include "dmp/Dmp.hpp"
#include "dmp/Trajectory.hpp"



class DmpExecutor
{
    public:
    Eigen::VectorXd y_des;
    Eigen::VectorXd yd_des;
    Eigen::VectorXd ydd_des;

    Eigen::VectorXd y_state;
    Eigen::VectorXd yd_state;
    Eigen::VectorXd ydd_state;

    Eigen::Vector2d zmp;
    Eigen::Vector3d ee_pos;
    Eigen::Vector3d lf_pos;
    Eigen::Vector3d rf_pos;
    double torso_z;
    double dt;

    Eigen::MatrixXd cost_vars;

    std::vector<std::string> names;;
    std::vector<std::string> left_arm_names;
    std::vector<std::string> right_arm_names;
    std::vector<std::string> head_names;
    std::vector<std::string> left_leg_names;
    std::vector<std::string> right_leg_names;
    std::vector<std::string> torso_names;

    // Publishers
    // ros::Publisher command_pub;
    ros::Publisher left_arm_pub;
    ros::Publisher right_arm_pub;
    ros::Publisher head_pub;
    ros::Publisher torso_pub;
    ros::Publisher left_leg_pub;
    ros::Publisher right_leg_pub;

    // Subscribers
    ros::Subscriber base_sub;
    ros::Subscriber zmp_sub;
    ros::Subscriber state_sub;

    // ros::Subscriber left_arm_sub;
    // ros::Subscriber right_arm_sub;
    // ros::Subscriber head_sub;
    // ros::Subscriber torso_sub;
    // ros::Subscriber left_leg_sub;
    // ros::Subscriber right_leg_sub;

    // tf2_ros::Buffer tfBuffer;
    // tf2_ros::TransformListener listener(tfBuffer);

    std::map<std::string, int> joint_names_map;

    void init();
    void base_callback(nav_msgs::Odometry msg);
    void zmp_callback(visualization_msgs::Marker msg);
    void state_callback(sensor_msgs::JointState msg);

    // sensor_msgs::JointState make_trj_to_point(Eigen::VectorXd positions, Eigen::VectorXd velocities, Eigen::VectorXd accelerations, double freq);
    trajectory_msgs::JointTrajectory make_trj_to_point(std::vector<std::string> limb_names, Eigen::VectorXd positions, Eigen::VectorXd velocities, Eigen::VectorXd accelerations, double freq);
    void integrateStep(double dt, Eigen::VectorXd y_des, Eigen::VectorXd yd_des, Eigen::VectorXd ydd_des);
    Eigen::VectorXd get_state(double t);

    void run(DmpBbo::Dmp dmp);

};