/*
 */

#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

#include <ros/ros.h>
// #include <trajectory_msgs/JointTrajectory.h>
// #include <trajectory_msgs/JointTrajectoryPoint.h>
// #include <control_msgs/JointTrajectoryControllerState.h>
// #include <sensor_msgs/JointState.h>
// #include <visualization_msgs/Marker.h>
// #include <tf2_ros/transform_listener.h>
// #include <tf2_ros/buffer.h>

#include <robotExecuteDmpReaching.hpp>

#include "dmp/Dmp.hpp"
#include "dmp/Trajectory.hpp"
#include "eigenutils/eigen_file_io.hpp"
#include "runSimulationThrowBall.hpp"

using namespace nlohmann;
using namespace std;
using namespace Eigen;
using namespace DmpBbo;

void help(char* binary_name)
{
  cout << "Usage: " << binary_name
       << " <dmp filename.json> <cost vars filename.txt>" << endl;
}


void DmpExecutor::init()
{
  // ROS 
  ros::NodeHandle nh;
//   command_pub = nh.advertise<sensor_msgs::JointState>("/command_state", 120);
  left_arm_pub = nh.advertise<trajectory_msgs::JointTrajectory>("/left_arm_controller/command", 120);
  right_arm_pub = nh.advertise<trajectory_msgs::JointTrajectory>("/right_arm_controller/command", 120);
  head_pub = nh.advertise<trajectory_msgs::JointTrajectory>("/head_controller/command", 120);
  torso_pub = nh.advertise<trajectory_msgs::JointTrajectory>("/torso_controller/command", 120);
  left_leg_pub = nh.advertise<trajectory_msgs::JointTrajectory>("/left_leg_controller/command", 120);
  right_leg_pub = nh.advertise<trajectory_msgs::JointTrajectory>("/right_leg_controller/command", 120);

  std::cout << "------------ 01 -----------------" << std::endl;

  
  base_sub = nh.subscribe("/floating_base_pose_simulated", 120, &DmpExecutor::base_callback, this, ros::TransportHints().reliable().tcpNoDelay());
  zmp_sub = nh.subscribe("/zmp", 120, &DmpExecutor::zmp_callback, this, ros::TransportHints().reliable().tcpNoDelay());
  state_sub = nh.subscribe("/joint_states", 120, &DmpExecutor::state_callback, this, ros::TransportHints().reliable().tcpNoDelay());
//   left_arm_sub = nh.subscribe("/left_arm_controller/state", 120, &DmpExecutor::state_callback, this, ros::TransportHints().reliable().tcpNoDelay());
//   right_arm_sub = nh.subscribe("/right_arm_controller/state", 120, &DmpExecutor::state_callback, this, ros::TransportHints().reliable().tcpNoDelay());
//   head_sub = nh.subscribe("/head_controller/state", 120, &DmpExecutor::state_callback, this, ros::TransportHints().reliable().tcpNoDelay());
//   torso_sub = nh.subscribe("/torso_controller/state", 120, &DmpExecutor::state_callback, this, ros::TransportHints().reliable().tcpNoDelay());
//   left_leg_sub = nh.subscribe("/left_leg_controller/state", 120, &DmpExecutor::state_callback, this, ros::TransportHints().reliable().tcpNoDelay());
//   right_leg_sub = nh.subscribe("/right_leg_controller/state", 120, &DmpExecutor::state_callback, this, ros::TransportHints().reliable().tcpNoDelay());

std::cout << "------------ 02 -----------------" << std::endl;
    y_state = Eigen::VectorXd::Zero(30);
    yd_state = Eigen::VectorXd::Zero(30);
    ydd_state = Eigen::VectorXd::Zero(30);
    
    names.resize(30);
    names = {"arm_left_1_joint", 
             "arm_left_1_joint",
             "arm_left_2_joint",
             "arm_left_3_joint",
             "arm_left_4_joint",
             "arm_left_5_joint",
             "arm_left_6_joint",
             "arm_left_7_joint",
             "arm_right_1_joint",
             "arm_right_2_joint",
             "arm_right_3_joint",
             "arm_right_4_joint",
             "arm_right_5_joint",
             "arm_right_6_joint",
             "arm_right_7_joint",
             "head_1_joint",
             "head_2_joint",
             "leg_left_1_joint",
             "leg_left_2_joint",
             "leg_left_3_joint",
             "leg_left_4_joint",
             "leg_left_5_joint",
             "leg_left_6_joint",
             "leg_right_1_joint",
             "leg_right_2_joint",
             "leg_right_3_joint",
             "leg_right_4_joint",
             "leg_right_5_joint",
             "leg_right_6_joint",
             "torso_1_joint",
             "torso_2_joint"};

    left_arm_names = {"arm_left_1_joint",
                      "arm_left_2_joint",
                      "arm_left_3_joint",
                      "arm_left_4_joint",
                      "arm_left_5_joint",
                      "arm_left_6_joint",
                      "arm_left_7_joint"};
    right_arm_names = {"arm_right_1_joint",
                       "arm_right_2_joint",
                       "arm_right_3_joint",
                       "arm_right_4_joint",
                       "arm_right_5_joint",
                       "arm_right_6_joint",
                       "arm_right_7_joint"};
    head_names = {"head_1_joint",
                  "head_2_joint"};
    left_leg_names = {"leg_left_1_joint",
                      "leg_left_2_joint",
                      "leg_left_3_joint",
                      "leg_left_4_joint",
                      "leg_left_5_joint",
                      "leg_left_6_joint"};
    right_leg_names = {"leg_right_1_joint",
                       "leg_right_2_joint",
                       "leg_right_3_joint",
                       "leg_right_4_joint",
                       "leg_right_5_joint",
                       "leg_right_6_joint"};
    torso_names = {"torso_1_joint",
                   "torso_2_joint"};

    for(size_t i=0; i<names.size(); i++)
    {
        joint_names_map[names[i]] = i;
    }
    std::cout << "------------ 03 -----------------" << std::endl;
}

void DmpExecutor::base_callback(nav_msgs::Odometry msg)
{
    torso_z = msg.pose.pose.position.z;
}

void DmpExecutor::zmp_callback(visualization_msgs::Marker msg)
{
    zmp[0] = msg.pose.position.x;
    zmp[1] = msg.pose.position.y;
}

void DmpExecutor::state_callback(sensor_msgs::JointState msg)
{
    for(size_t i=0; i<names.size(); i++)
    {
        if(names[i].find("gripper") == std::string::npos)
        {
            y_state[joint_names_map[names[i]]] = msg.position[i];
            yd_state[joint_names_map[names[i]]] = msg.velocity[i];
            ydd_state[joint_names_map[names[i]]] = msg.effort[i];
        }
    }
}

// sensor_msgs::JointState DmpExecutor::make_trj_to_point(Eigen::VectorXd positions, Eigen::VectorXd velocities, Eigen::VectorXd accelerations, double freq)
// {
//     // std::vector<std::vector<std::string>> limbs = {left_arm_names, right_arm_names, head_names, left_leg_names, right_leg_names, torso_names};
//     // std::map<std::string, trajectory_msgs::JointTrajectory> trajectories;

//     sensor_msgs::JointState command_state;
//     command_state.header.stamp = ros::Time().now();
//     command_state.name = names;
//     for(unsigned i =0; i<positions.size(); i++)
//     {
//         command_state.position.push_back(positions(i));
//         command_state.velocity.push_back(velocities(i));
//         command_state.effort.push_back(accelerations(i));
//     }


//     // for(size_t n=0; n<limbs.size();n++)
//     // {
//     //     std::vector<std::string> limb_names = limbs[n];
//     //     trajectory_msgs::JointTrajectory trj;
//     //     trj.header.stamp = ros::Time().now();
//     //     trj.joint_names.resize(limb_names.size());
//     //     trj.joint_names = limb_names;
//     //     trajectory_msgs::JointTrajectoryPoint jtp;
//     //     if (freq > 1)
//     //         jtp.time_from_start.fromNSec(int(10e9*(1/freq)));
//     //     else
//     //         jtp.time_from_start.fromSec(int(1/freq));
//     //     for(size_t i =0; i < limb_names.size(); i++)
//     //     {
//     //         jtp.positions.push_back(positions(joint_names_map[limb_names[i]]));// = positions;
//     //         jtp.velocities.push_back(velocities(joint_names_map[limb_names[i]]));// = velocities;
//     //         jtp.accelerations.push_back(accelerations(joint_names_map[limb_names[i]]));// = accelerations;
//     //     }
//     //      trj.points.push_back(jtp);

//     //     trajectories[limb_names[0].substr(0,-7)] = trj;
//     // }

//     return command_state;

// }

trajectory_msgs::JointTrajectory DmpExecutor::make_trj_to_point(std::vector<std::string> limb_names, Eigen::VectorXd positions, Eigen::VectorXd velocities, Eigen::VectorXd accelerations, double freq)
{

    trajectory_msgs::JointTrajectory trj;
    std::cout << "------------ 0771 -----------------" << std::endl;
    trj.header.stamp = ros::Time().now();
    trj.joint_names.resize(limb_names.size()); 
    trj.joint_names = limb_names;


    trajectory_msgs::JointTrajectoryPoint jtp;
    if (freq > 1)
        jtp.time_from_start.fromNSec(int(10e9*(1/freq)));
    else
        jtp.time_from_start.fromSec(int(1/freq));
    std::cout << "------------ 0772 -----------------" << std::endl;
    
    for(size_t n =0; n<limb_names.size(); n++)
    {
        jtp.positions.push_back(positions(joint_names_map[limb_names[n]]));// = positions;
        jtp.velocities.push_back(velocities(joint_names_map[limb_names[n]]));// = velocities;
        jtp.accelerations.push_back(accelerations(joint_names_map[limb_names[n]]));// = accelerations;
    }
    trj.points.push_back(jtp);
    std::cout << "------------ 0773 -----------------" << std::endl;
    std::cout << trj.joint_names[0] << std::endl;

    return trj;
}

void DmpExecutor::integrateStep(double dt, Eigen::VectorXd y_des, Eigen::VectorXd yd_des, Eigen::VectorXd ydd_des)
{
    std::cout << "names = " << names[0] << std::endl;

    // std::cout << "------------ 071 -----------------" << std::endl;
    // std::vector<double> left_arm_pos;
    // std::vector<double> right_arm_pos;
    // std::vector<double> head_pos;
    // std::vector<double> left_leg_pos;
    // std::vector<double> right_leg_pos;
    // std::vector<double> torso_pos;

    // std::vector<double> left_arm_vel;
    // std::vector<double> right_arm_vel;
    // std::vector<double> head_vel;
    // std::vector<double> left_leg_vel;
    // std::vector<double> right_leg_vel;
    // std::vector<double> torso_vel;
    
    // std::vector<double> left_arm_acc;
    // std::vector<double> right_arm_acc;
    // std::vector<double> head_acc;
    // std::vector<double> left_leg_acc;
    // std::vector<double> right_leg_acc;
    // std::vector<double> torso_acc;

    
    // for(size_t i=0; i<7; i++)
    // {

    //     left_arm_pos.push_back(y_des(i));
    //     left_arm_vel.push_back(yd_des(i));
    //     left_arm_acc.push_back(ydd_des(i));

    //     std::cout << y_des(i) << std::endl;
    //     std::cout << yd_des(i) << std::endl;
    //     std::cout << ydd_des(i) << std::endl;
    //     std::cout << y_des(7+i) << std::endl;
    //     std::cout << yd_des(7+i) << std::endl;
    //     std::cout << ydd_des(7+i) << std::endl;
    //     right_arm_pos.push_back(y_des(7+i));
    //     right_arm_vel.push_back(yd_des(7+i));
    //     right_arm_acc.push_back(ydd_des(7+i));
    // }
    // std::cout << "------------ 072 -----------------" << std::endl;

    // for(size_t i=0; i<6; i++)
    // {
    //     left_leg_pos.push_back(y_des(16+i));
    //     left_leg_vel.push_back(yd_des(16+i));
    //     left_leg_acc.push_back(ydd_des(16+i));

    //     right_leg_pos.push_back(y_des(22+i));
    //     right_leg_vel.push_back(yd_des(22+i));
    //     right_leg_acc.push_back(ydd_des(22+i));
    // }

    // for(size_t i=0; i<2; i++)
    // {
    //     head_pos.push_back(y_des(14+i));
    //     head_vel.push_back(yd_des(14+i));
    //     head_acc.push_back(ydd_des(14+i));

    //     torso_pos.push_back(y_des(28+i));
    //     torso_vel.push_back(yd_des(28+i));
    //     torso_acc.push_back(ydd_des(28+i));
    // }


    // VectorXd::Map(&left_arm_pos[0], 7) = y_des.segment(0,7);
    // VectorXd::Map(&right_arm_pos[0], 7) = y_des.segment(7,7);
    // VectorXd::Map(&head_pos[0], 2) = y_des.segment(14,2);
    // VectorXd::Map(&left_leg_pos[0], 6) = y_des.segment(16,6);
    // VectorXd::Map(&right_leg_pos[0], 6) = y_des.segment(22,6);
    // VectorXd::Map(&torso_pos[0], 6) = y_des.segment(28,2);
    // std::cout << "------------ 073 -----------------" << std::endl;

    // std::cout << "------------ 074 -----------------" << std::endl;
    
    // VectorXd::Map(&left_arm_vel[0], 7) = yd_des.segment(0,7);
    // VectorXd::Map(&right_arm_vel[0], 7) = yd_des.segment(7,7);
    // VectorXd::Map(&head_vel[0], 2) = yd_des.segment(14,2);
    // VectorXd::Map(&left_leg_vel[0], 6) = yd_des.segment(16,6);
    // VectorXd::Map(&right_leg_vel[0], 6) = yd_des.segment(22,6);
    // VectorXd::Map(&torso_vel[0], 6) = yd_des.segment(28,2);
    // std::cout << "------------ 075 -----------------" << std::endl;


    // std::cout << "------------ 076 -----------------" << std::endl;
    
    // VectorXd::Map(&left_arm_acc[0], 7) = ydd_des.segment(0,7);
    // VectorXd::Map(&right_arm_acc[0], 7) = ydd_des.segment(7,7);
    // VectorXd::Map(&head_acc[0], 2) = ydd_des.segment(14,2);
    // VectorXd::Map(&left_leg_acc[0], 6) = ydd_des.segment(16,6);
    // VectorXd::Map(&right_leg_acc[0], 6) = ydd_des.segment(22,6);
    // VectorXd::Map(&torso_acc[0], 6) = ydd_des.segment(28,2);

    // std::cout << y_des.size() << std::endl;
    // std::cout << yd_des.size() << std::endl;
    // std::cout << ydd_des.size() << std::endl;
    // std::cout << left_leg_names[0] << std::endl;
    // std::cout << right_leg_names[0] << std::endl;
    // std::cout << torso_names[0] << std::endl;
    // std::cout << 1/dt << std::endl;

    
    // sensor_msgs::JointState command_state = make_trj_to_point(y_des, yd_des, ydd_des, 1/dt);
    trajectory_msgs::JointTrajectory left_arm_traj =  make_trj_to_point(left_arm_names, y_des, yd_des, ydd_des, 1/dt);
    trajectory_msgs::JointTrajectory right_arm_traj = make_trj_to_point(right_arm_names, y_des, yd_des, ydd_des, 1/dt);
    trajectory_msgs::JointTrajectory head_traj =      make_trj_to_point(head_names, y_des, yd_des, ydd_des, 1/dt);
    trajectory_msgs::JointTrajectory left_leg_traj =  make_trj_to_point(left_leg_names, y_des, yd_des, ydd_des, 1/dt);
    trajectory_msgs::JointTrajectory right_leg_traj = make_trj_to_point(right_leg_names, y_des, yd_des, ydd_des, 1/dt);
    trajectory_msgs::JointTrajectory torso_traj =     make_trj_to_point(torso_names, y_des, yd_des, ydd_des, 1/dt);
    std::cout << "------------ 078 -----------------" << std::endl;


    // command_pub.publish(command_state);
    ros::Duration(0.0083333).sleep(); 
    left_arm_pub.publish(left_arm_traj);
    right_arm_pub.publish(right_arm_traj);
    head_pub.publish(head_traj);
    left_leg_pub.publish(left_leg_traj);
    right_leg_pub.publish(right_leg_traj);
    torso_pub.publish(torso_traj);

    // left_arm_pub.publish(trajs["arm_left"]);
    // right_arm_pub.publish(trajs["arm_right"]);
    // head_pub.publish(trajs["head"]);
    // left_leg_pub.publish(trajs["leg_left"]);
    // right_leg_pub.publish(trajs["leg_right"]);
    // torso_pub.publish(trajs["torso"]);
    std::cout << "------------ 079 -----------------" << std::endl;  
}

Eigen::VectorXd DmpExecutor::get_state(double t)
{
    Eigen::VectorXd state(1+90+2+3+3+3);
    state(0) = t;
    state.segment(1,30) = y_state;
    state.segment(31,30) = yd_state;
    state.segment(61,30) = ydd_state;
    state.segment(91,2) = zmp;
    state.segment(93,3) = ee_pos;
    state.segment(96,3) = lf_pos;
    state.segment(99,3) = rf_pos;
    return state;
}

void DmpExecutor::run(DmpBbo::Dmp dmp)
{
    // Same time vector as in the demonstration trajectory
    // auto ts = dmp.tau();
    // int n_time_steps = ts.size();

    // Integrate DMP longer than the tau with which it was trained

    double integration_time = 1.5 * dmp.tau();
    dt = 0.008333333;
    std::cout << "dt = " << dt << std::endl;
    std::cout << "integration time = " << integration_time << std::endl;
    

    int n_time_steps = floor(integration_time / dt);
    std::cout << "ntimesteps = " << n_time_steps << std::endl;

    cost_vars.resize(n_time_steps, 1 + 90 + 2 + 3 + 3 + 3);
    std::cout << "------------ 05 -----------------" << std::endl;

    // Run simulation
    // dmp.integrateStart(y_state, yd_state);
    double time = 0;
    std::cout << "------------ 06 -----------------" << std::endl;

    for (int ii = 0; ii < n_time_steps; ii++) 
    {
        time += dt;
        // double dt = ts[ii+1] - ts[ii];
        dmp.stateAsPosVelAcc(y_state, yd_state, y_des, yd_des, ydd_des);
        std::cout << "------------ 07 -----------------" << std::endl;
        integrateStep(dt, y_des, yd_des, ydd_des);
        std::cout << "------------ 08 -----------------" << std::endl;
        cost_vars.row(ii) = get_state(time);
        std::cout << "------------ 09 -----------------" << std::endl;
        dmp.integrateStep(dt, y_state, y_state, yd_state);
        std::cout << "------------ 10 -----------------" << std::endl;
    }
}

/** Main function
 * \param[in] n_args Number of arguments
 * \param[in] args Arguments themselves
 * \return Success of exection. 0 if successful.
 */
int main(int n_args, char** args)
{
  if (n_args != 3) {
    help(args[0]);
    return -1;
  }

  if (string(args[1]).compare("--help") == 0) {
    help(args[0]);
    return 0;
  }

  string dmp_filename = string(args[1]);
  string cost_vars_filename = string(args[2]);

  cout << "C++ Reading Dmp <-   " << dmp_filename << endl;

  // Load DMP
  ifstream file(dmp_filename);
  json j = json::parse(file);
  DmpBbo::Dmp* dmp = j.get<DmpBbo::Dmp*>();
//   int argc;
//   char** argv;
  std::cout << "------------ 1 -----------------" << std::endl;
  
  ros::init(n_args, args, "dmp_executor");
  DmpExecutor dmp_executor;
  dmp_executor.init();
  dmp_executor.run(*dmp);
  std::cout << "------------ 2 -----------------" << std::endl;

  // Save cost_vars to file
    bool overwrite = true;
    cout << "C++ Writing   -> " << cost_vars_filename << endl;
    saveMatrix("./", cost_vars_filename, dmp_executor.cost_vars, overwrite);

    delete dmp;

  return 0;
}
