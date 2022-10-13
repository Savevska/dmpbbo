import os
from posixpath import join
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from dmpbbo.bbo_of_dmps.Task import Task

class TaskReach(Task):
    
    def __init__(self, ee_pos_goal, pos_margin, ref_cop, stability_weight, goal_weight, traj_weight, traj_demonstrated):

        self.ee_pos_goal_ = ee_pos_goal
        self.pos_margin_ = pos_margin
        self.ref_cop_ = [0.0, 0.0]
        self.traj_demonstrated_ = traj_demonstrated
        self.stability_weight_ = stability_weight
        self.goal_weight_ = goal_weight
        self.traj_weight_ = traj_weight
    
    def costLabels(self):
        return ['stability', 'goal reached', 'trajectory follow']

    def evaluate_rollout(self,cost_vars,sample):
        # self.ref_cop_ = np.loadtxt("/home/ksavevska/catkin_ws/src/rgbd_motion_imitation/src/zmp_ref.txt")
        # print("Reference CoP", self.ref_cop_)
        self.ee_pos_goal_ = [0.65, -0.4, 0.0] 
        n_dims = 30#self.traj_demonstrated_.dim_#n_dims = 13
        n_misc = 11#self.traj_demonstrated_.dim_misc()#n_misc = 5
        n_time_steps = cost_vars.shape[0]
        
        ts = cost_vars[:,0]
        joint_states = cost_vars[:,1:n_dims+1]
        cop_x = cost_vars[:,-n_misc]
        cop_y = cost_vars[:,-n_misc+1]
        
        ee_pos_x = cost_vars[:,-n_misc+2]
        ee_pos_y = cost_vars[:,-n_misc+3]
        ee_pos_z = cost_vars[:,-n_misc+4]
        
        # self.ref_cop_[0] = cost_vars[0, -n_misc+5]
        # self.ref_cop_[1] = cost_vars[0, -n_misc+6]

        # print("Reference CoP", self.ref_cop_)


        # stability cost 
        # TODO: add margins
        ########### 1
        # fall_penalty = 1
        # x_limit = 0.125016 # not sure if I should take the 
        # y_limit = 0.149483
        # steepness_x = fall_penalty/(x_limit**3)
        # steepness_y = fall_penalty/(y_limit**3)
        # dist_x = (steepness_x*np.abs(cop_x - self.ref_cop_[0]))**3
        # dist_y = (steepness_y*np.abs(cop_y - self.ref_cop_[1]))**3
        # dist_to_ref_cop = dist_x + dist_y
        
        ########### 2
        # dist_to_ref_cop = np.sqrt((cop_x - self.ref_cop_[0])**2 + (cop_y-self.ref_cop_[1])**2)

        ########### 3
        x_size = 0.21
        y_size = 0.13
        self.sp_y_1 = cost_vars[0, -2] - y_size/2
        self.sp_y_2 = cost_vars[0, -5] + y_size/2
        self.sp_x_1 = max(cost_vars[0, -6] - x_size/2, cost_vars[0, -3] - x_size/2)
        self.sp_x_2 = min(cost_vars[0, -6] + x_size/2, cost_vars[0, -3] + x_size/2)
        
        x_cost = abs(((2*cop_x - self.sp_x_1 - self.sp_x_2)/(self.sp_x_2 - self.sp_x_1))**3)
        y_cost = abs(((2*cop_y - self.sp_y_1 - self.sp_y_2)/(self.sp_y_2 - self.sp_y_1))**3)
        stability_cost = 0.6*x_cost + 0.4*y_cost

        # dist_to_ref_cop = (1/len(cost_vars))*np.sqrt((cop_x - self.ref_cop_[0])**2 + (cop_y-self.ref_cop_[1])**2)

        stability_cost = [0]*len(cop_x)
        for i in range(len(cop_x)):
            if cop_x[i] > self.sp_x_1 and cop_x[i] < self.sp_x_2 and cop_y[i] > self.sp_y_1 and cop_y[i] < self.sp_y_2:
                if min(abs(self.sp_x_1-cop_x[i]), abs(self.sp_x_2-cop_x[i]), abs(self.sp_y_1-cop_y[i]), abs(self.sp_y_2-cop_y[i])) > 0.0095:
                    stability_cost[i] = (min(x_size/2, (self.sp_y_2-self.sp_y_1)/2) / (min(abs(self.sp_x_1-cop_x[i]), abs(self.sp_x_2-cop_x[i]), abs(self.sp_y_1-cop_y[i]), abs(self.sp_y_2-cop_y[i]))) - 1)**2
                else:
                    stability_cost[i] = 100
            else:
                stability_cost[i] = 100
        stability_cost = np.array(stability_cost)
        # goal cost (euclidean distance from the goal ee position)
        # TODO: add margins
        # dist_to_goal = np.sqrt((ee_pos_x - self.ee_pos_goal_[0])**2 + (ee_pos_y-self.ee_pos_goal_[1])**2 + (ee_pos_z-self.ee_pos_goal_[2])**2)
        dist_to_goal = (np.sqrt((ee_pos_x[-1] - self.ee_pos_goal_[0])**2 + (ee_pos_y[-1]-self.ee_pos_goal_[1])**2 + (ee_pos_z[-1]-self.ee_pos_goal_[2])**2))
        # trajectory cost (difference between the demonstrated trajectory and executed trajectory)
        # TODO: add margins
        dist_to_traj = np.sum([abs(joint_states[i] - self.traj_demonstrated_._ys[i]) for i in range(len(self.traj_demonstrated_._ys))], axis=0)

        # costs sum
        costs = np.zeros(1+3)
        costs[1] = np.sum(self.stability_weight_*stability_cost)
        costs[2] = np.sum(self.goal_weight_*dist_to_goal)
        costs[3] = np.sum(self.traj_weight_*dist_to_traj)

        costs[0] = np.sum(costs[1:])
        
        return costs
        
    def plot_rollout(self,cost_vars,ax):
        """Simple script to plot y of DMP trajectory"""

        n_dims = 24#self.traj_demonstrated_.dim_
        n_misc = 5#self.traj_demonstrated_.dim_misc()

        t = cost_vars[:,0]
        y = cost_vars[:,1:n_dims]
        misc = cost_vars[:,-n_misc:]

        # fig1, ((ax11, ax12), (ax21, ax22), (ax31, ax32), (ax41, ax42)) = plt.subplots(nrows= 4, ncols=2, num=1, clear=True)
        # ax11.plot(y[:,1], label='left_1')
        # ax21.plot(y[:,2], label='left_2')
        # ax31.plot(y[:,4], label='left_4')
        # ax41.plot(y[:,12], label='left_leg_2')
        # ax12.plot(y[:,5], label='right_1')
        # ax22.plot(y[:,6], label='right_2')
        # ax32.plot(y[:,8], label='right_4')
        # ax42.plot(y[:,18], label='right_leg_2')
        # fig1.legend()
        # fig1.show()

        fig=plt.figure()
        gs = GridSpec(2,2, figure=fig)

        ax1 = fig.add_subplot(gs[0,0])
        ax1.plot(misc[:,0], label="COP X", color='blue')

        ax2 = fig.add_subplot(gs[1,0])
        ax2.plot(misc[:,1], label="COP Y", color="red")

        ax3 = fig.add_subplot(gs[:, 1])
        ax3.plot(misc[:,0], misc[:,1], label="COP", color="orange")

        fig.legend()
        fig.show()
        
        # line_handles = ax.plot(y[:,0],y[:,1],linewidth=0.5)
        # line_handles_ball_traj = ax.plot(ball[:,0],ball[:,1],'-')
        # line_handles_ball = ax.plot(ball[::5,0],ball[::5,1],'ok')
        # plt.setp(line_handles_ball,'MarkerFaceColor','none')

        # line_handles.extend(line_handles_ball_traj)

        # # Plot the floor
        # x_floor = [-1.0, self.x_goal_-self.x_margin_,self.x_goal_, self.x_goal_+self.x_margin_ ,0.4]
        # y_floor = [self.y_floor_,self.y_floor_,self.y_floor_-0.05,self.y_floor_,self.y_floor_]
        # ax.plot(x_floor,y_floor,'-k',linewidth=1)
        # ax.plot(self.x_goal_,self.y_floor_-0.05,'og')
        # ax.axis('equal')
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_xlim([-0.9, 0.3])
        # ax.set_ylim([-0.4, 0.3])
            
        # return line_handles
