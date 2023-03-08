'''
 * \file robotExecuteDmp.cpp
 * \author Freek Stulp
 *
 * \ingroup Demos
 *
 * This file is part of DmpBbo, a set of libraries and programs for the
 * black-box optimization of dynamical movement primitives.
 * Copyright (C) 2014 Freek Stulp, ENSTA-ParisTech
 *
 * DmpBbo is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * DmpBbo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with DmpBbo.  If not, see <http://www.gnu.org/licenses/>.
'''
import numpy as np
import sys

sys.path.append("/home/ksavevska/dmpbbo")
import argparse
# from dmpbbo.dmps import Dmp
# from dmpbbo.dmps import Trajectory
import dmpbbo.json_for_cpp as jc

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from sensor_msgs.msg import JointState
from rqt_joint_trajectory_controller import joint_limits_urdf

from geometry_msgs.msg import Pose, PoseArray
# from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker
import tf2_ros
from tf.transformations import euler_from_quaternion
import quaternion

# import copy
import os


# import time

class DmpExecution:
  def __init__(self, args):
    
    self.dmp_rarm_filename = args.dmp_rarm_filename
    self.dmp_larm_filename = args.dmp_larm_filename
    self.cost_vars_rarm_filename = args.cost_vars_rarm_filename
    self.cost_vars_larm_filename = args.cost_vars_larm_filename  

    self.rate = rospy.Rate(120)
    # self.last_pub_time = time.time()
    # self.min_pub_dt = 1 /1200

    self.tfBuffer = tf2_ros.Buffer()
    self.listener = tf2_ros.TransformListener(self.tfBuffer)

    print("Reading right arm DMP <-   ", args.dmp_rarm_filename)
    self.dmp_rarm = jc.loadjson(args.dmp_rarm_filename) # json parser to the dmp object
    print("Reading left arm DMP <-   ", args.dmp_larm_filename)
    self.dmp_larm = jc.loadjson(args.dmp_larm_filename) # json parser to the dmp object

    print(self.dmp_rarm.tau)
    print(self.dmp_rarm.dim_y)

    # same time vector
    # self.ts = self.dmp_rarm.ts_train
    # longer time vector
    tau_exec = 1.3 * self.dmp_rarm.tau
    dt = 1/120 
    n_time_steps = int(tau_exec / dt)
    self.ts = np.linspace(0, tau_exec, n_time_steps, dtype=float)  

    msg_rarm = rospy.wait_for_message(topic="/right_arm_controller/state", topic_type=JointTrajectoryControllerState, timeout=1)
    # msg_larm = rospy.wait_for_message(topic="/left_arm_controller/state", topic_type=JointTrajectoryControllerState, timeout=1)

    
    # pos = np.delete(np.array(msg.position), [14, 15])
    # vel = np.delete(np.array(msg.position), [14, 15])

    self.y_state = np.array([msg_rarm.actual.positions])
    self.yd_state = np.array([msg_rarm.actual.velocities])
    # self.y_state = np.column_stack([msg_rarm.actual.positions, msg_larm.actual.positions])
    # self.yd_state = np.column_stack([msg_rarm.actual.velocities, msg_larm.actual.velocities])
    self.ydd_state = np.zeros((1, 7))

    self.ee_pos = np.array([[0.0, 0.0, 0.0]])
    # self.ee_rpy = np.array([[0.0, 0.0, 0.0]])
    self.ee_rot = np.array([[0.0, 0.0, 0.0, 1.0]])


    self.lf_pos = np.array([[0.0, 0.0, 0.0]])
    self.rf_pos = np.array([[0.0, 0.0, 0.0]])

    self.zmp = np.array([[0.0, 0.0]])
    self.torso_z = 1.0

    self.cost_vars_cols = self.y_state.shape[1] + self.yd_state.shape[1] + self.ydd_state.shape[1] + self.zmp.shape[1] + self.ee_pos.shape[1] + self.ee_rot.shape[1] + self.lf_pos.shape[1] + self.rf_pos.shape[1]

    queue_size = 10
    self.left_arm_pub = rospy.Publisher("/left_arm_controller/command", JointTrajectory, queue_size=queue_size, tcp_nodelay=True)
    self.right_arm_pub = rospy.Publisher("/right_arm_controller/command", JointTrajectory, queue_size=queue_size, tcp_nodelay=True)
    self.head_pub = rospy.Publisher("/head_controller/command", JointTrajectory, queue_size=queue_size, tcp_nodelay=True)
    self.torso_pub = rospy.Publisher("/torso_controller/command", JointTrajectory, queue_size=queue_size, tcp_nodelay=True)
    self.left_leg_pub = rospy.Publisher("/left_leg_controller/command", JointTrajectory, queue_size=queue_size, tcp_nodelay=True)
    self.right_leg_pub = rospy.Publisher("/right_leg_controller/command", JointTrajectory, queue_size=queue_size, tcp_nodelay=True)

    # self.command_pub = rospy.Publisher("/whole_body_kinematic_controller/reference_ref", JointState, queue_size=120)

    self.right_pose_pub = rospy.Publisher("/right_wrist_pose", Pose, queue_size=queue_size, tcp_nodelay=True)
    self.left_pose_pub = rospy.Publisher("/left_wrist_pose", Pose, queue_size=queue_size, tcp_nodelay=True)

    self.poses_pub = rospy.Publisher("/wrist_poses", PoseArray, queue_size=queue_size, tcp_nodelay=True)

    # base_sub = rospy.Subscriber("/floating_base_pose_simulated", Odometry, self.base_callback, queue_size=queue_size, tcp_nodelay=True)
    torzo_z_sub = rospy.Subscriber("/torso_z", Float32, self.torso_z_callback, queue_size=queue_size, tcp_nodelay=True)

    zmp_sub = rospy.Subscriber("/cop", Marker, self.zmp_callback, queue_size=queue_size, tcp_nodelay=True)

    # self.left_arm_sub = rospy.Subscriber("/left_arm_controller/state", JointTrajectoryControllerState, self.state_callback, queue_size=queue_size)
    self.right_arm_sub = rospy.Subscriber("/right_arm_controller/state", JointTrajectoryControllerState, self.state_callback, queue_size=queue_size)
    # self.head_sub = rospy.Subscriber("/head_controller/state", JointTrajectoryControllerState, self.state_callback, queue_size=queue_size)
    # self.torso_sub = rospy.Subscriber("/torso_controller/state", JointTrajectoryControllerState, self.state_callback, queue_size=queue_size)
    # self.left_leg_sub = rospy.Subscriber("/left_leg_controller/state", JointTrajectoryControllerState, self.state_callback, queue_size=queue_size)
    # self.right_leg_sub = rospy.Subscriber("/right_leg_controller/state", JointTrajectoryControllerState, self.state_callback, queue_size=queue_size)

    # self.state_sub = rospy.Subscriber("/joint_states", JointState, self.state_callback, queue_size=queue_size, tcp_nodelay=True)

    self.left_arm_names =  ["arm_left_1_joint", "arm_left_2_joint", "arm_left_3_joint", "arm_left_4_joint", "arm_left_5_joint", "arm_left_6_joint", "arm_left_7_joint"]
    self.right_arm_names = ["arm_right_1_joint", "arm_right_2_joint", "arm_right_3_joint", "arm_right_4_joint", "arm_right_5_joint", "arm_right_6_joint", "arm_right_7_joint"]
    self.head_names =      ["head_1_joint", "head_2_joint"]
    self.left_leg_names =  ["leg_left_1_joint", "leg_left_2_joint", "leg_left_3_joint", "leg_left_4_joint", "leg_left_5_joint", "leg_left_6_joint"]
    self.right_leg_names = ["leg_right_1_joint", "leg_right_2_joint", "leg_right_3_joint", "leg_right_4_joint", "leg_right_5_joint", "leg_right_6_joint"]
    self.torso_names =     ["torso_1_joint", "torso_2_joint"]

    # self.names = ["arm_left_1_joint", "arm_left_2_joint", "arm_left_3_joint", "arm_left_4_joint", "arm_left_5_joint", "arm_left_6_joint", "arm_left_7_joint",\
    #               "arm_right_1_joint", "arm_right_2_joint", "arm_right_3_joint", "arm_right_4_joint", "arm_right_5_joint", "arm_right_6_joint", "arm_right_7_joint",\
    #               "head_1_joint", "head_2_joint", \
    #               "leg_left_1_joint", "leg_left_2_joint", "leg_left_3_joint", "leg_left_4_joint", "leg_left_5_joint", "leg_left_6_joint", \
    #               "leg_right_1_joint", "leg_right_2_joint", "leg_right_3_joint", "leg_right_4_joint", "leg_right_5_joint", "leg_right_6_joint", \
    #               "torso_1_joint", "torso_2_joint"]
  
  def base_callback(self, msg):
    self.torso_z = msg.pose.pose.position.z

  def torso_z_callback(self, msg):
    self.torso_z = msg.data

  def zmp_callback(self, msg):
    self.zmp[0][:] = [msg.pose.position.x, msg.pose.position.y]

  def state_callback(self, msg):
    # JointState message
    # pos = np.delete(np.array(msg.position), [14, 15])
    # vel = np.delete(np.array(msg.velocity), [14, 15])
    # self.ydd_state = np.array((vel - self.yd_state)*120)
    # self.y_state[:] = pos#np.array([pos])
    # self.yd_state[:] = vel# np.array([vel])

    # JointTrajectoryControllerState message
    # if "arm_left" in msg.joint_names[0]:
    #   self.y_state[0][0:7] = np.array([msg.actual.positions])
    #   self.yd_state[0][0:7] = np.array([msg.actual.velocities])
    #   self.ydd_state[0][0:7] = np.array([msg.actual.velocities]) + self.yd_state[0][0:7]
    # if "arm_right" in msg.joint_names[0]:
    self.y_state = np.array([msg.actual.positions])
    self.yd_state = np.array([msg.actual.velocities])
    self.ydd_state = -np.array([msg.actual.velocities]) + self.yd_state
    # if "head" in msg.joint_names[0]:
    #   self.y_state[0][14:16] = np.array([msg.actual.positions])
    #   self.yd_state[0][14:16] = np.array([msg.actual.velocities])
    #   self.ydd_state[0][14:16] = -np.array([msg.actual.velocities]) + self.yd_state[0][14:16]
    # if "leg_left" in msg.joint_names[0]:
    #   self.y_state[0][16:22] = np.array([msg.actual.positions])
    #   self.yd_state[0][16:22] = np.array([msg.actual.velocities])
    #   self.ydd_state[0][16:22] = -np.array([msg.actual.velocities]) + self.yd_state[0][16:22]
    # if "leg_right" in msg.joint_names[0]:
    #   self.y_state[0][22:28] = np.array([msg.actual.positions])
    #   self.yd_state[0][22:28] = np.array([msg.actual.velocities])
    #   self.ydd_state[0][22:28] = -np.array([msg.actual.velocities]) + self.yd_state[0][22:28]
    # if "torso" in msg.joint_names[0]:
    #   self.y_state[0][28:30] = np.array([msg.actual.positions])
    #   self.yd_state[0][28:30] = np.array([msg.actual.velocities])
    #   self.ydd_state[0][28:30] = -np.array([msg.actual.velocities]) + self.yd_state[0][28:30]
  
  def make_trj_to_point(self, names, positions, velocities, accelerations, freq):
      '''Make message for moving to a cetrian joint configuration'''

      # trajectory msg
      trj = JointTrajectory()
      trj.header.stamp = rospy.Time.now()
      trj.joint_names = names

      # point towards which we interpolate
      jtp = JointTrajectoryPoint()

      # timeout to get to the desired configuration
      if freq > 1:
          jtp.time_from_start.nsecs = int(10e9*1/freq)#time_from_start
      else:
          jtp.time_from_start.secs = int(1/freq)
      # joint configurations in rad and rad/s
      jtp.positions = positions
      jtp.velocities = velocities
      jtp.accelerations = accelerations

      trj.points = [jtp]

      return trj 
    
  def integrateStep(self, dt, pos_rarm, rot_rarm, pos_larm, rot_larm):
    pose_rarm = Pose()
    pose_rarm.position.x = pos_rarm[0]
    pose_rarm.position.y = pos_rarm[1]
    pose_rarm.position.z = pos_rarm[2]

    rot_rarm = quaternion.as_float_array(rot_rarm)
    rot_rarm_norm = rot_rarm / np.linalg.norm(rot_rarm)
    pose_rarm.orientation.w = rot_rarm_norm[0]
    pose_rarm.orientation.x = rot_rarm_norm[1]
    pose_rarm.orientation.y = rot_rarm_norm[2]
    pose_rarm.orientation.z = rot_rarm_norm[3]    

    pose_larm = Pose()
    pose_larm.position.x = pos_larm[0]
    pose_larm.position.y = pos_larm[1]
    pose_larm.position.z = pos_larm[2]

    rot_larm = quaternion.as_float_array(rot_larm)
    rot_larm_norm = rot_larm / np.linalg.norm(rot_larm)
    pose_larm.orientation.w = rot_larm_norm[0]
    pose_larm.orientation.x = rot_larm_norm[1]
    pose_larm.orientation.y = rot_larm_norm[2]
    pose_larm.orientation.z = rot_larm_norm[3]    
    
    # self.right_pose_pub.publish(pose_rarm)
    # self.left_pose_pub.publish(pose_larm)
    
    poses = PoseArray()
    poses.poses = [pose_rarm, pose_larm]
    self.poses_pub.publish(poses)

    self.rate.sleep()

  def failed(self):
    if self.torso_z < 0.5:
      return True
    else:
      return False
  
  def get_state(self, dt):
    try:
      self.trans = self.tfBuffer.lookup_transform("odom", 'wrist_right_ft_link', rospy.Time(), rospy.Duration(dt))
      self.trans_left = self.tfBuffer.lookup_transform("odom", 'left_sole_link', rospy.Time(), rospy.Duration(dt))
      self.trans_right = self.tfBuffer.lookup_transform("odom", 'right_sole_link', rospy.Time(), rospy.Duration(dt))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        self.rate.sleep()

    self.ee_pos[0][:] = [self.trans.transform.translation.x, self.trans.transform.translation.y, self.trans.transform.translation.z]
    # print(self.trans.transform.rotation)
    # self.ee_rpy[0][:] = euler_from_quaternion([self.trans.transform.rotation.x, self.trans.transform.rotation.y, self.trans.transform.rotation.z, self.trans.transform.rotation.w], axes="rxyz")
    self.ee_rot[0][:] = [self.trans.transform.rotation.x, self.trans.transform.rotation.y, self.trans.transform.rotation.z, self.trans.transform.rotation.w]

    self.lf_pos[0][:] = [self.trans_left.transform.translation.x, self.trans_left.transform.translation.y, self.trans_left.transform.translation.z]
    
    self.rf_pos[0][:] = [self.trans_right.transform.translation.x, self.trans_right.transform.translation.y, self.trans_right.transform.translation.z]


    # self.ee_pos = np.array([[self.trans.transform.translation.x, self.trans.transform.translation.y, self.trans.transform.translation.z]])
    # self.lf_pos = np.array([[self.trans_left.transform.translation.x, self.trans_left.transform.translation.y, self.trans_left.transform.translation.z]])
    # self.rf_pos = np.array([[self.trans_right.transform.translation.x, self.trans_right.transform.translation.y, self.trans_right.transform.translation.z]])

    costs = np.column_stack((self.y_state, self.yd_state, self.ydd_state, self.zmp, self.ee_pos, self.ee_rot, self.lf_pos, self.rf_pos))
    return costs[0].tolist()
  
  def reset_pose(self):
    rospy.sleep(3)
    freq = 0.2
    left_arm_traj =      self.make_trj_to_point(self.left_arm_names, [0.3, 0.4, -0.5, -1.5, 0.0, 0.0, 0.0], [], [], freq)
    right_arm_traj =     self.make_trj_to_point(self.right_arm_names, [-0.3, -0.4, 0.5, -1.5, 0.0, 0.0, 0.0], [], [], freq)
    head_traj =          self.make_trj_to_point(self.head_names, [0.0, 0.0], [], [], freq)
    left_leg_traj =      self.make_trj_to_point(self.left_leg_names, [0.0, 0.0, -0.4, 0.8, -0.4, 0.0], [], [], freq)
    right_leg_traj =     self.make_trj_to_point(self.right_leg_names, [0.0, 0.0, -0.4, 0.8, -0.4, 0.0], [], [], freq)
    torso_traj =         self.make_trj_to_point(self.torso_names, [0.0, 0.0], [], [], freq)

    self.left_arm_pub.publish(left_arm_traj)
    self.right_arm_pub.publish(right_arm_traj)
    self.head_pub.publish(head_traj)
    self.left_leg_pub.publish(left_leg_traj)
    self.right_leg_pub.publish(right_leg_traj)
    self.torso_pub.publish(torso_traj)
    rospy.sleep(5)
    if self.failed():
      self.cost_vars[-1][:] = (100*np.ones(len(self.cost_vars[-1][:]))).tolist()
      os.system("rosservice call /gazebo/reset_world && sleep 8")
  
  def run(self):
    self.cost_vars = []
    rot_rarm_des = np.quaternion(self.dmp_rarm.q0[0], self.dmp_rarm.q0[1], self.dmp_rarm.q0[2], self.dmp_rarm.q0[3])
    z_rarm = np.quaternion(0,0,0,0)
    x_phase_rarm, _ = self.dmp_rarm._phase_system_rot.integrate_start()
    x_rarm, xd_rarm = self.dmp_rarm.integrate_start()

    rot_larm_des = np.quaternion(self.dmp_larm.q0[0], self.dmp_larm.q0[1], self.dmp_larm.q0[2], self.dmp_larm.q0[3])
    z_larm = np.quaternion(0,0,0,0)
    x_phase_larm, _ = self.dmp_larm._phase_system_rot.integrate_start()
    x_larm, xd_larm = self.dmp_larm.integrate_start()
    
    for i in range(len(self.ts)-1):
      if not self.failed():
        dt = self.ts[i+1] - self.ts[i]
        pos_rarm_des, vel_rarm_des, acc_rarm_des = self.dmp_rarm.states_as_pos_vel_acc(x_rarm, xd_rarm)
        pos_larm_des, vel_larm_des, acc_larm_des = self.dmp_larm.states_as_pos_vel_acc(x_larm, xd_larm)

        self.integrateStep(dt, pos_rarm_des, rot_rarm_des, pos_larm_des, rot_larm_des)
              
        costs = self.get_state(dt)
        self.cost_vars.append(costs)
        x_phase_rarm, rot_rarm_des, z_rarm = self.dmp_rarm.integrate_step_quaternion(x_phase_rarm, rot_rarm_des, z_rarm, dt)
        x_phase_larm, rot_larm_des, z_larm = self.dmp_larm.integrate_step_quaternion(x_phase_larm, rot_larm_des, z_larm, dt)

        x_rarm, xd_rarm = self.dmp_rarm.integrate_step_euler(dt, x_rarm)
        x_larm, xd_larm = self.dmp_larm.integrate_step_euler(dt, x_larm)

      else:
        print("Robot failed!")
        costs_fail = np.ones((len(self.ts)-i, self.cost_vars_cols))*100
        if len(self.cost_vars) >= 1:
          self.cost_vars = np.row_stack((np.array(self.cost_vars), costs_fail)).tolist()
        else:
          self.cost_vars = costs_fail.tolist()
        break
    self.reset_pose()
    np.savetxt(fname=self.cost_vars_rarm_filename, X=np.array(self.cost_vars))
    np.savetxt(fname=self.cost_vars_larm_filename, X=np.array(self.cost_vars))

      


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("dmp_rarm_filename", help="file to read dmp from")
  parser.add_argument("dmp_larm_filename", help="file to read dmp from")
  parser.add_argument("cost_vars_rarm_filename", help="directory to write cost relevant data")
  parser.add_argument("cost_vars_larm_filename", help="directory to write cost relevant data")

  args = parser.parse_args()
  rospy.init_node("execute_dmp", anonymous=True)
  dmpExecutor = DmpExecution(args)
  dmpExecutor.run()

