import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from pytransform3d.rotations import quaternion_integrate, matrix_from_quaternion, plot_basis

def calculate_dist_to_cop(cost_vars):                        
    n_misc = 15                                                                                         
    cop_x = cost_vars[:,-n_misc]                                  
    cop_y = cost_vars[:,-n_misc+1]                                   

    rf_y = cost_vars[0, -2]                                                            
    lf_y = cost_vars[0, -5]                       
    rf_x = cost_vars[0, -3]
    lf_x = cost_vars[0, -6]             
    ref_cop = np.array([(rf_x + lf_x)/2, (rf_y + lf_y)/2])
    dist_to_cop = np.sum((1/len(cop_x))*np.sqrt((cop_x - ref_cop[0])**2 + (cop_y - ref_cop[1])**2))
    return dist_to_cop 

def plot_costs(results_folder):
    updates = np.sort(os.listdir(os.path.join(results_folder, "updates_rarm")))
    with open(results_folder + '/plot_configs.json') as config_file:
        plot_configs = json.load(config_file)
    
    costs = []
    dist_to_cop = []
    for update in updates:
        if "update0" in update and update != "update00150":
            cost_files = np.sort(os.listdir(os.path.join(results_folder, "updates_rarm/"+update)))
            for cost in cost_files:
                if "eval_costs.txt" in cost:
                    c = np.loadtxt(os.path.join(results_folder+"/updates_rarm/"+update, cost))
                    costs.append(c)
                    cv = np.loadtxt(results_folder+"/updates_rarm/"+update+"/eval_cost_vars.txt")
                    d=calculate_dist_to_cop(cv)                                                    
                    dist_to_cop.append(d)
    res = pd.DataFrame(columns=["cost", "stab_cost", "goal_cost", "orient_cost", "acc_cost", "traj_cost"], data=costs)
    res["goal_cost"] = dist_to_cop

    weights = plot_configs["weights"]
    titles = plot_configs["titles"]
    for col in res.columns:                                      
        plt.plot(res[col]/weights[col])                                                                     
        plt.title(titles[col])                                        
        plt.grid()
        plt.show() 
        plt.savefig(results_folder+"/"+titles[col]+".pdf")
                                                           
    return res

def plot_ee_traj(results_folder):
    cost_vars_0 = np.loadtxt(results_folder + "/updates_rarm/update00000/eval_cost_vars.txt")
    cost_vars_T = np.loadtxt(results_folder + "/updates_rarm/update00149/eval_cost_vars.txt")
    n_misc = 15
    ee_pos_x_0 = cost_vars_0[:,-n_misc+2]
    ee_pos_y_0 = cost_vars_0[:,-n_misc+3]
    ee_pos_z_0 = cost_vars_0[:,-n_misc+4]
    pos_0 = np.column_stack((ee_pos_x_0, ee_pos_y_0, ee_pos_z_0))
    ee_rot_x_0 = cost_vars_0[:,-n_misc+5]
    ee_rot_y_0 = cost_vars_0[:,-n_misc+6]
    ee_rot_z_0 = cost_vars_0[:,-n_misc+7]
    ee_rot_w_0 = cost_vars_0[:,-n_misc+8]
    Q_0 = np.column_stack((ee_rot_w_0, ee_rot_x_0, ee_rot_y_0, ee_rot_z_0))
    
    ee_pos_x_T = cost_vars_T[:,-n_misc+2]
    ee_pos_y_T = cost_vars_T[:,-n_misc+3]
    ee_pos_z_T = cost_vars_T[:,-n_misc+4]
    pos_T = np.column_stack((ee_pos_x_T, ee_pos_y_T, ee_pos_z_T))
    ee_rot_x_T = cost_vars_T[:,-n_misc+5]
    ee_rot_y_T = cost_vars_T[:,-n_misc+6]
    ee_rot_z_T = cost_vars_T[:,-n_misc+7]
    ee_rot_w_T = cost_vars_T[:,-n_misc+8]
    Q_T = np.column_stack((ee_rot_w_T, ee_rot_x_T, ee_rot_y_T, ee_rot_z_T))
    
    fig3d = plt.figure()
    ax = Axes3D(fig3d)
    ax.view_init(None, 220)
    n=0
    ax.plot(ee_pos_x_0[n:], ee_pos_y_0[n:], ee_pos_z_0[n:], linewidth=1.5, label="initial")
    ax.plot(ee_pos_x_T[n:], ee_pos_y_T[n:], ee_pos_z_T[n:], linewidth=1.5, label="optimal")
    ax.scatter(0.65, -0.4, 0.00, s=100, color="green")
    for i in [0,-1]:
        R_0 = matrix_from_quaternion(Q_0[i])
        p_0 = pos_0[i]
        R_T = matrix_from_quaternion(Q_T[i])
        p_T = pos_T[i]
        ax = plot_basis(ax=ax, s=0.1, R=R_0, p=p_0)
        ax = plot_basis(ax=ax, s=0.1, R=R_T, p=p_T)
    ax = plot_basis(ax=ax, s=0.1, R=matrix_from_quaternion([0.5, 0.5, -0.5, -0.5]), p=[0.65,-0.4, 0.0])

    ax.legend()
    ax.set_title("EE trajectory comparison")
    # plt.savefig(results_folder+"/ee_traj_comparison.pdf")
    plt.show()


if __name__ == "__main__":
    results_folder = sys.args[1]
    
    plot_costs(results_folder)
    plot_ee_traj(results_folder)

