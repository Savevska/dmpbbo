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
import os
import sys
# sys.path.append("/home/ksavevska/dmpbbo")
sys.path.append("/home/user/talos_ws/dmpbbo")
import argparse
# from dmpbbo.dmps import Dmp
# from dmpbbo.dmps import Trajectory
import dmpbbo.json_for_cpp as jc

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from sensor_msgs.msg import JointState

# from geometry_msgs.msg import Point
# from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker
import tf2_ros
from tf.transformations import euler_from_quaternion
# from tf2_msgs.msg import TFMessage
# import copy
# import os
# from filters import KalmanFilter


# import time

class DmpExecution:
  def __init__(self, args):
    
    self.dmp_filename = args.dmp_filename
    self.cost_vars_filename = args.cost_vars_filename


    self.rate = rospy.Rate(120)
    # self.last_pub_time = time.time()
    # self.min_pub_dt = 1 /1200

    self.tfBuffer = tf2_ros.Buffer()
    self.listener = tf2_ros.TransformListener(self.tfBuffer)

    print("Reading DMP <-   ", args.dmp_filename)
    self.dmp = jc.loadjson(args.dmp_filename) # json parser to the dmp object

    print(self.dmp.tau)
    print(self.dmp.dim_y)

    # same time vector
    self.ts = self.dmp.ts_train
    # longer time vector
    # tau_exec = 3.5 * self.dmp.tau
    # dt = 1/120 
    # n_time_steps = int(tau_exec / dt)
    # self.ts = np.linspace(0, tau_exec, n_time_steps, dtype=float)  

    msg = rospy.wait_for_message(topic="/joint_states", topic_type=JointState, timeout=0.1)
    # msg_larm = rospy.wait_for_message(topic="/left_arm_controller/state", topic_type=JointTrajectoryControllerState, timeout=1)
    # msg_rarm = rospy.wait_for_message(topic="/right_arm_controller/state", topic_type=JointTrajectoryControllerState, timeout=1)
    # msg_head = rospy.wait_for_message(topic="/head_controller/state", topic_type=JointTrajectoryControllerState, timeout=1)
    # msg_torso = rospy.wait_for_message(topic="/torso_controller/state", topic_type=JointTrajectoryControllerState, timeout=1)

    # pos = np.delete(np.array(msg.position), [14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29])
    # vel = np.delete(np.array(msg.position), [14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29])
    self.ind = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 30, 31])

    self.y_state = np.array(msg.position)[self.ind].reshape(1,-1)
    self.yd_state = np.array(msg.velocity)[self.ind].reshape(1,-1)
    self.ydd_state = np.zeros((1,self.dmp.dim_y))

    self.y_des = np.zeros((1,self.dmp.dim_y))
    self.yd_des = np.zeros((1,self.dmp.dim_y))
    self.ydd_des = np.zeros((1,self.dmp.dim_y))

    # self.vel_kf = np.array([KalmanFilter(R=10e-4, init_val=self.yd_state[0][i]) for i in range(self.yd_state.shape[1])])
    # self.acc_kf = np.array([KalmanFilter(R=10e-4, init_val=self.ydd_state[0][i]) for i in range(self.ydd_state.shape[1])])

    # self.y_state = np.zeros((1,self.dmp.dim_y))
    # self.yd_state = np.zeros((1,self.dmp.dim_y))
    # self.ydd_state = np.zeros((1,self.dmp.dim_y))

    # self.y_state[:7] = msg_larm.actual.positions
    # self.y_state[7:14] = msg_rarm.actual.positions
    # self.y_state[14:16] = msg_head.actual.positions
    # self.y_state[16:18] = msg_torso.actual.positions

    # self.y_state[:7] = msg_larm.actual.velocities
    # self.y_state[7:14] = msg_rarm.actual.velocities
    # self.y_state[14:16] = msg_head.actual.velocities
    # self.y_state[16:18] = msg_torso.actual.velocities


    self.ee_pos = np.array([[0.0, 0.0, 0.0]])
    # self.ee_rpy = np.array([[0.0, 0.0, 0.0]])
    self.ee_rot = np.array([[0.0, 0.0, 0.0, 1.0]])

    self.lf_pos = np.array([[0.0, 0.0, 0.0]])
    self.rf_pos = np.array([[0.0, 0.0, 0.0]])

    self.zmp = np.array([[0.0, 0.0]])
    self.torso_z = 1.0

    self.cost_vars_cols = 2*self.y_state.shape[1] + 2*self.yd_state.shape[1] + 2*self.ydd_state.shape[1] + self.zmp.shape[1] + self.ee_pos.shape[1] + self.ee_rot.shape[1] + self.lf_pos.shape[1] + self.rf_pos.shape[1]


    queue_size = 10
    self.left_arm_pub = rospy.Publisher("/left_arm_controller/command", JointTrajectory, queue_size=queue_size, tcp_nodelay=True)
    self.right_arm_pub = rospy.Publisher("/right_arm_controller/command", JointTrajectory, queue_size=queue_size, tcp_nodelay=True)
    self.head_pub = rospy.Publisher("/head_controller/command", JointTrajectory, queue_size=queue_size, tcp_nodelay=True)
    self.torso_pub = rospy.Publisher("/torso_controller/command", JointTrajectory, queue_size=queue_size, tcp_nodelay=True)
    self.left_leg_pub = rospy.Publisher("/left_leg_controller/command", JointTrajectory, queue_size=queue_size, tcp_nodelay=True)
    self.right_leg_pub = rospy.Publisher("/right_leg_controller/command", JointTrajectory, queue_size=queue_size, tcp_nodelay=True)

    # self.command_pub = rospy.Publisher("/whole_body_kinematic_controller/reference_ref", JointState, queue_size=120)

    # base_sub = rospy.Subscriber("/floating_base_pose_simulated", Odometry, self.base_callback, queue_size=queue_size, tcp_nodelay=True)
    torzo_z_sub = rospy.Subscriber("/torso_z", Float32, self.torzo_z_callback, queue_size=queue_size, tcp_nodelay=True)

    zmp_sub = rospy.Subscriber("/cop", Marker, self.zmp_callback, queue_size=queue_size, tcp_nodelay=True)

    # self.left_arm_sub = rospy.Subscriber("/left_arm_controller/state", JointTrajectoryControllerState, self.state_callback, queue_size=queue_size, tcp_nodelay=True)
    # self.right_arm_sub = rospy.Subscriber("/right_arm_controller/state", JointTrajectoryControllerState, self.state_callback, queue_size=queue_size, tcp_nodelay=True)
    # self.head_sub = rospy.Subscriber("/head_controller/state", JointTrajectoryControllerState, self.state_callback, queue_size=queue_size, tcp_nodelay=True)
    # self.torso_sub = rospy.Subscriber("/torso_controller/state", JointTrajectoryControllerState, self.state_callback, queue_size=queue_size, tcp_nodelay=True)
    # self.left_leg_sub = rospy.Subscriber("/left_leg_controller/state", JointTrajectoryControllerState, self.state_callback, queue_size=queue_size)
    # self.right_leg_sub = rospy.Subscriber("/right_leg_controller/state", JointTrajectoryControllerState, self.state_callback, queue_size=queue_size)

    self.state_sub = rospy.Subscriber("/joint_states", JointState, self.state_callback, queue_size=queue_size, tcp_nodelay=True)

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
    while(self.left_arm_pub.get_num_connections() == 0 and \
          self.right_arm_pub.get_num_connections() == 0 and \
          self.head_pub.get_num_connections() == 0 and \
          self.torso_pub.get_num_connections() == 0 and \
          self.left_leg_pub.get_num_connections() == 0 and \
          self.right_leg_pub.get_num_connections() == 0):
        rospy.sleep(0.01)
  
  def base_callback(self, msg):
    self.torso_z = msg.pose.pose.position.z

  def torzo_z_callback(self, msg):
    self.torso_z = msg.data

  def zmp_callback(self, msg):
    self.zmp[0][:] = [msg.pose.position.x, msg.pose.position.y]

  def state_callback(self, msg):
    # JointState message
    # pos = np.delete(np.array(msg.position), [14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29])
    # vel = np.delete(np.array(msg.velocity), [14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29])
    # self.ydd_state = np.array((self.yd_state - vel))
    # self.y_state[:] = pos#np.array([pos])
    # self.yd_state[:] = vel# np.array([vel])
    
    # Without filtering
    self.ydd_state = np.array((-self.yd_state + np.array(msg.velocity)[self.ind]))*120
    self.yd_state = np.array(msg.velocity)[self.ind].reshape(1,-1)
    self.y_state = np.array(msg.position)[self.ind].reshape(1,-1)

    # With filtering
    # self.ydd_state = np.array((-self.yd_state + np.array((-self.y_state + np.array(msg.position)[self.ind]))*1000))*1000
    # self.yd_state = np.array((-self.y_state + np.array(msg.position)[self.ind]))*1000
    # for i in range(len(self.yd_state)):
    #   self.yd_state[0][i] = self.vel_kf[i].filt(self.yd_state[0][i])
    #   self.ydd_state[0][i] = self.acc_kf[i].filt(self.ydd_state[0][i])
    # self.y_state = np.array(msg.position)[self.ind].reshape(1,-1)




    # JointTrajectoryControllerState message
    # if "arm_left" in msg.joint_names[0]:
    #   self.ydd_state[0][0:7] = -np.array([msg.actual.velocities]) + self.yd_state[0][0:7]
    #   self.y_state[0][0:7] = np.array([msg.actual.positions])
    #   self.yd_state[0][0:7] = np.array([msg.actual.velocities])
    # if "arm_right" in msg.joint_names[0]:
    #   self.ydd_state[0][7:14] = -np.array([msg.actual.velocities]) + self.yd_state[0][7:14]
    #   self.y_state[0][7:14] = np.array([msg.actual.positions])
    #   self.yd_state[0][7:14] = np.array([msg.actual.velocities])
    # if "head" in msg.joint_names[0]:
    #   self.ydd_state[0][14:16] = -np.array([msg.actual.velocities]) + self.yd_state[0][14:16]
    #   self.y_state[0][14:16] = np.array([msg.actual.positions])
    #   self.yd_state[0][14:16] = np.array([msg.actual.velocities])
    # if "leg_left" in msg.joint_names[0]:
    #   self.y_state[0][16:22] = np.array([msg.actual.positions])
    #   self.yd_state[0][16:22] = np.array([msg.actual.velocities])
    #   self.ydd_state[0][16:22] = -np.array([msg.actual.velocities]) + self.yd_state[0][16:22]
    # if "leg_right" in msg.joint_names[0]:
    #   self.y_state[0][22:28] = np.array([msg.actual.positions])
    #   self.yd_state[0][22:28] = np.array([msg.actual.velocities])
    #   self.ydd_state[0][22:28] = -np.array([msg.actual.velocities]) + self.yd_state[0][22:28]
    # if "torso" in msg.joint_names[0]:
    #   self.ydd_state[0][16:18] = -np.array([msg.actual.velocities]) + self.yd_state[0][16:18]
    #   self.y_state[0][16:18] = np.array([msg.actual.positions])
    #   self.yd_state[0][16:18] = np.array([msg.actual.velocities])

  def make_trj_to_point(self, names, positions, velocities, accelerations, freq):
      '''Make message for moving to a cetrian joint configuration'''

      # trajectory msg
      trj = JointTrajectory()
      # trj.header.stamp = rospy.Time.now()
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
      # jtp.velocities = velocities
      # jtp.accelerations = accelerations

      trj.points = [jtp]

      return trj      

  def integrateStep(self, dt, y_des, yd_des, ydd_des):

    left_arm_pos =      y_des[0:7]
    right_arm_pos =     y_des[7:14]
    head_pos =          y_des[14:16]
    # left_leg_pos =      y_des[16:22]
    # right_leg_pos =     y_des[22:28]
    # torso_pos =         y_des[28:30]
    torso_pos =         y_des[16:18]
    
    left_arm_vel =      yd_des[0:7]
    right_arm_vel =     yd_des[7:14]
    head_vel =          yd_des[14:16]
    # left_leg_vel =      yd_des[16:22]
    # right_leg_vel =     yd_des[22:28]
    # torso_vel =         yd_des[28:30]
    torso_vel =         yd_des[16:18]


    left_arm_acc =      ydd_des[0:7]
    right_arm_acc =     ydd_des[7:14]
    head_acc =          ydd_des[14:16]
    # left_leg_acc =      ydd_des[16:22]
    # right_leg_acc =     ydd_des[22:28]
    # torso_acc =         ydd_des[28:30]
    torso_acc =         ydd_des[16:18]


    left_arm_traj =      self.make_trj_to_point(self.left_arm_names, left_arm_pos, left_arm_vel, left_arm_acc, 1/dt)
    right_arm_traj =     self.make_trj_to_point(self.right_arm_names, right_arm_pos, right_arm_vel, right_arm_acc, 1/dt)
    head_traj =          self.make_trj_to_point(self.head_names, head_pos, head_vel, head_acc, 1/dt)
    # left_leg_traj =      self.make_trj_to_point(self.left_leg_names, left_leg_pos, left_leg_vel, left_leg_acc, 1/dt)
    # right_leg_traj =     self.make_trj_to_point(self.right_leg_names, right_leg_pos, right_leg_vel, right_leg_acc, 1/dt)
    left_leg_traj =      self.make_trj_to_point(self.left_leg_names, [0.0,  0.0, -0.26,  0.6, -0.33, 0.0], [0.0]*6, [0.0]*6, 1/dt)
    right_leg_traj =     self.make_trj_to_point(self.right_leg_names, [0.0,  0.0, -0.26,  0.6 , -0.33, 0.0], [0.0]*6, [0.0]*6, 1/dt)
    torso_traj =         self.make_trj_to_point(self.torso_names, torso_pos, torso_vel, torso_acc, 1/dt)

    self.rate.sleep()
    self.left_arm_pub.publish(left_arm_traj)
    self.right_arm_pub.publish(right_arm_traj)
    self.head_pub.publish(head_traj)
    self.left_leg_pub.publish(left_leg_traj)
    self.right_leg_pub.publish(right_leg_traj)
    self.torso_pub.publish(torso_traj)

  # def integrateStep(self, dt, y_des, yd_des, ydd_des):
  #   command_msg = JointState()
  #   command_msg.name = self.names
  #   command_msg.position = y_des
  #   command_msg.velocity = yd_des
  #   command_msg.effort = [0]*len(y_des)

  #   # command.acceleration = ydd_des

  #   self.command_pub.publish(command_msg)
    # rospy.sleep(dt)

  def failed(self):
    if self.torso_z < 0.5:
      return True
    else:
      return False
  
  def get_state(self, dt):
    try:
      self.trans = self.tfBuffer.lookup_transform("base_link", 'wrist_right_ft_link',rospy.Time(0))
      self.trans_left = self.tfBuffer.lookup_transform("odom", 'left_sole_link', rospy.Time(0))
      self.trans_right = self.tfBuffer.lookup_transform("odom", 'right_sole_link', rospy.Time(0))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        self.rate.sleep()

    self.ee_pos[0][:] = [self.trans.transform.translation.x, self.trans.transform.translation.y, self.trans.transform.translation.z]
    # self.ee_rpy[0][:] = euler_from_quaternion(self.trans.transform.rotation)
    self.ee_rot[0][:] = [self.trans.transform.rotation.x, self.trans.transform.rotation.y, self.trans.transform.rotation.z, self.trans.transform.rotation.w]

    self.lf_pos[0][:] = [self.trans_left.transform.translation.x, self.trans_left.transform.translation.y, self.trans_left.transform.translation.z]
    
    self.rf_pos[0][:] = [self.trans_right.transform.translation.x, self.trans_right.transform.translation.y, self.trans_right.transform.translation.z]

    # tf_test = np.array([[self.tf_msg.transform.translation.x, self.tf_msg.transform.translation.y, self.tf_msg.transform.translation.z]])
    # self.ee_pos = np.array([[self.trans.transform.translation.x, self.trans.transform.translation.y, self.trans.transform.translation.z]])
    # self.lf_pos = np.array([[self.trans_left.transform.translation.x, self.trans_left.transform.translation.y, self.trans_left.transform.translation.z]])
    # self.rf_pos = np.array([[self.trans_right.transform.translation.x, self.trans_right.transform.translation.y, self.trans_right.transform.translation.z]])

    costs = np.column_stack((self.y_state, self.yd_state, self.ydd_state, np.array([self.y_des]), np.array([self.yd_des]), np.array([self.ydd_des]), self.zmp, self.ee_pos, self.ee_rot, self.lf_pos, self.rf_pos))
    return costs[0].tolist()
  
  def reset_pose(self):
    rospy.sleep(3)
    freq = 0.2
    # left_arm_traj =      self.make_trj_to_point(self.left_arm_names, [0.3, 0.4, -0.5, -1.5, 0.0, 0.0, 0.0], [], [], freq)
    # right_arm_traj =     self.make_trj_to_point(self.right_arm_names, [-0.3, -0.4, 0.5, -1.5, 0.0, 0.0, 0.0], [], [], freq)
    # head_traj =          self.make_trj_to_point(self.head_names, [0.0, 0.0], [], [], freq)
    # # left_leg_traj =      self.make_trj_to_point(self.left_leg_names, [0.0, 0.0, -0.4, 0.8, -0.4, 0.0], [], [], freq)
    # # right_leg_traj =     self.make_trj_to_point(self.right_leg_names, [0.0, 0.0, -0.4, 0.8, -0.4, 0.0], [], [], freq)
    # torso_traj =         self.make_trj_to_point(self.torso_names, [0.0, 0.0], [], [], freq)

    left_arm_traj =      self.make_trj_to_point(self.left_arm_names, [0.6,  0.3, -0.5, -0.6,  0.0,  0.0, 0.002], [], [], freq)
    right_arm_traj =     self.make_trj_to_point(self.right_arm_names, [-0.6, -0.3, 0.5, -0.6, 0.0, 0.0, 0.002], [], [], freq)
    head_traj =          self.make_trj_to_point(self.head_names, [0.0, 0.0], [], [], freq)
    # left_leg_traj =      self.make_trj_to_point(self.left_leg_names, [0.0,  0.0, -0.2,  0.5, -0.28, 0.0], [], [], freq)
    # right_leg_traj =     self.make_trj_to_point(self.right_leg_names, [0.0,  0.0, -0.2,  0.5 , -0.28, 0.0], [], [], freq)
    left_leg_traj =      self.make_trj_to_point(self.left_leg_names, [0.0,  0.0, -0.26,  0.6, -0.33, 0.0], [], [], freq)
    right_leg_traj =     self.make_trj_to_point(self.right_leg_names, [0.0,  0.0, -0.26,  0.6 , -0.33, 0.0], [], [], freq)
    torso_traj =         self.make_trj_to_point(self.torso_names, [0.0,  0.25], [], [], freq)

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
    x, xd = self.dmp.integrate_start()
    for i in range(len(self.ts)):
      if not self.failed():
        if i < (len(self.ts)-1):
          dt = self.ts[i+1] - self.ts[i]
        else:
          dt = self.ts[-1] - self.ts[-2]
        # dt = self.ts[i+1] - self.ts[i]
        self.y_des, self.yd_des, self.ydd_des = self.dmp.states_as_pos_vel_acc(x, xd)
        self.integrateStep(dt, self.y_des, self.yd_des, self.ydd_des)
              
        costs = self.get_state(dt)
        self.cost_vars.append(costs)
        x, xd = self.dmp.integrate_step_euler(dt, x)
      else:
        print("Robot failed!")
        costs_fail = np.ones((len(self.ts)-i, self.cost_vars_cols))*100
        if len(self.cost_vars) >= 1:
          self.cost_vars = np.row_stack((np.array(self.cost_vars), costs_fail)).tolist()
        else:
          self.cost_vars = costs_fail.tolist()
        break 
    self.reset_pose()
    np.savetxt(fname=self.cost_vars_filename, X=np.array(self.cost_vars))
      


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("dmp_filename", help="file to read dmp from")
  parser.add_argument("cost_vars_filename", help="directory to write cost relevant data")
  args = parser.parse_args()
  rospy.init_node("execute_dmp", anonymous=True)
  dmpExecutor = DmpExecution(args)
  dmpExecutor.run()

