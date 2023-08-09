# This file is part of DmpBbo, a set of libraries and programs for the
# black-box optimization of dynamical movement primitives.
# Copyright (C) 2022 Freek Stulp
#
# DmpBbo is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# DmpBbo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DmpBbo.  If not, see <http://www.gnu.org/licenses/>.
""" Script for training a DMP from a trajectory. """


import argparse
import os
from pathlib import Path
import sys
# sys.path.append("/home/ksavevska/dmpbbo")
sys.path.append("/home/user/talos_ws/dmpbbo")

# sys.path.append("/Users/kristina/WORK/dmpbbo")
import numpy as np
from matplotlib import pyplot as plt

import dmpbbo.json_for_cpp as jc
from dmpbbo.dmps.CartDmp import CartDmp
from dmpbbo.dmps.Trajectory import Trajectory
from dmpbbo.functionapproximators.FunctionApproximatorRBFN import FunctionApproximatorRBFN
from dmpbbo.dynamicalsystems.SigmoidSystem import SigmoidSystem

import quaternion


def main():
    
    """ Main function that is called when executing the script. """

    parser = argparse.ArgumentParser()
    parser.add_argument("rarm_trajectory_file", help="file to read trajectory from")
    parser.add_argument("larm_trajectory_file", help="file to read trajectory from")
    parser.add_argument("output_directory", help="directory to write dmp and other results to")
    parser.add_argument("--n", help="max number of basis functions", type=int, default=15)
    parser.add_argument("--show", action="store_true", help="Show plots")
    parser.add_argument("--save", action="store_true", help="save result plots to png")
    args = parser.parse_args()

    os.makedirs(args.output_directory, exist_ok=True)

    ################################################
    # Read trajectory and train DMP with it.

    print(f"Reading right arm trajectory from: {args.rarm_trajectory_file}\n")
    tr = np.loadtxt(args.rarm_trajectory_file)
    print(f"Reading left arm trajectory from: {args.larm_trajectory_file}\n")
    tl = np.loadtxt(args.larm_trajectory_file)


    traj_rarm = Trajectory(ts=tr[:,0], ys=tr[:,1:])
    filename_traj_rarm = Path(args.output_directory, "trajectory_rarm.txt")
    traj_rarm.savetxt(filename_traj_rarm)

    traj_larm = Trajectory(ts=tl[:,0], ys=tl[:,1:])
    filename_traj_larm = Path(args.output_directory, "trajectory_larm.txt")
    traj_larm.savetxt(filename_traj_larm)
    # jc.savejson(traj,Path(args.output_directory,'trajectory.json'))
    n_dims = traj_rarm.dim
    print("Dims = ", n_dims)
    traj_rarm._misc = None
    traj_larm._misc = None

    peak_to_peak_rarm = np.ptp(traj_rarm.ys, axis=0)  # Range of data; used later on
    peak_to_peak_larm = np.ptp(traj_larm.ys, axis=0)  # Range of data; used later on

    mean_absolute_errors_rarm = []
    mean_absolute_errors_larm = []

    n_bfs_list = list(range(args.n, args.n + 1))
    for n_bfs in n_bfs_list:

        rarm_function_apps = [FunctionApproximatorRBFN(n_bfs, 0.7) for _ in range(3)]
        rarm_function_apps_rot = [FunctionApproximatorRBFN(n_bfs, 0.7) for _ in range(3)]

        larm_function_apps = [FunctionApproximatorRBFN(n_bfs, 0.7) for _ in range(3)]
        larm_function_apps_rot = [FunctionApproximatorRBFN(n_bfs, 0.7) for _ in range(3)]


        dmp_rarm = CartDmp.from_traj(traj_rarm, rarm_function_apps, rarm_function_apps_rot, dmp_type="KULVICIUS_2012_JOINING", gating_system=SigmoidSystem(tau=traj_rarm.duration, x_init=1, max_rate=-1.0, inflection_ratio=0.9))
        dmp_larm = CartDmp.from_traj(traj_larm, larm_function_apps, larm_function_apps_rot, dmp_type="KULVICIUS_2012_JOINING", gating_system=SigmoidSystem(tau=traj_larm.duration, x_init=1, max_rate=-1.0, inflection_ratio=0.9))


        # These are the parameters that will be optimized.
        dmp_rarm.set_selected_param_names(["weights", "goal"])
        # dmp_rarm.set_selected_param_names(["weights"])
        dmp_larm.set_selected_param_names(["weights"])

        ################################################
        # Save DMP to file
        d = args.output_directory
        
        filename_rarm = Path(d, f"dmp_rarm_trained_{n_bfs}.json")
        print(f"Saving trained DMP for right arm to: {filename_rarm}")
        jc.savejson(filename_rarm, dmp_rarm)
        jc.savejson_for_cpp(Path(d, f"dmp_rarm_trained_{n_bfs}_for_cpp.json"), dmp_rarm)

        filename_larm = Path(d, f"dmp_larm_trained_{n_bfs}.json")
        print(f"Saving trained DMP for left arm to: {filename_larm}")
        jc.savejson(filename_larm, dmp_larm)
        jc.savejson_for_cpp(Path(d, f"dmp_larm_trained_{n_bfs}_for_cpp.json"), dmp_larm)

        ################################################
        # Analytical solution to compute difference

        ts_rarm = traj_rarm.ts
        xs_ana_rarm, xds_ana_rarm, _, _, q_traj_rarm = dmp_rarm.analytical_solution(ts_rarm)
        traj_reproduced_ana_rarm = dmp_rarm.states_as_trajectory(ts_rarm, xs_ana_rarm, xds_ana_rarm)
        cart_traj_reproduced_ana_rarm = np.column_stack((traj_reproduced_ana_rarm.ys, q_traj_rarm))

        mae_rarm = np.mean(abs(traj_rarm.ys - cart_traj_reproduced_ana_rarm))
        mean_absolute_errors_rarm.append(mae_rarm)
        print()
        print(f"               Number of basis functions: {n_bfs}")
        print(f"MAE between demonstration and reproduced: {mae_rarm}")
        print(f"                           Range of data: {peak_to_peak_rarm}")
        print()

        ts_larm = traj_larm.ts
        xs_ana_larm, xds_ana_larm, _, _, q_traj_larm = dmp_larm.analytical_solution(ts_larm)
        traj_reproduced_ana_larm = dmp_larm.states_as_trajectory(ts_larm, xs_ana_larm, xds_ana_larm)
        cart_traj_reproduced_ana_larm = np.column_stack((traj_reproduced_ana_larm.ys, q_traj_larm))

        mae_larm = np.mean(abs(traj_larm.ys - cart_traj_reproduced_ana_larm))
        mean_absolute_errors_larm.append(mae_larm)
        print()
        print(f"               Number of basis functions: {n_bfs}")
        print(f"MAE between demonstration and reproduced: {mae_larm}")
        print(f"                           Range of data: {peak_to_peak_larm}")
        print()

        ################################################
        # Integrate DMP (with 1.3*tau)

        # tau_exec = 1.3 * traj.duration
        # dt = 1/120 
        # n_time_steps = int(tau_exec / dt)
        # ts = np.zeros([n_time_steps, 1])
        # xs_step = np.zeros([n_time_steps, dmp_rarm.dim_x])
        # xds_step = np.zeros([n_time_steps, dmp_rarm.dim_x])

        # x, xd = dmp_rarm.integrate_start()
        # xs_step[0, :] = x
        # xds_step[0, :] = xd
        # for tt in range(1, n_time_steps):
        #     ts[tt] = dt * tt
        #     xs_step[tt, :], xds_step[tt, :] = dmp_rarm.integrate_step(dt, xs_step[tt - 1, :])

        # traj_reproduced = dmp_rarm.states_as_trajectory(ts, xs_step, xds_step)

       
        ##################### RIGHT ARM
        # Integrate DMP (with same tau) 
        n_time_steps = len(traj_rarm._ts)
        ts = traj_rarm._ts
        # Integrate DMP (with 1.3*tau)
        # tau_exec = 1.3 * traj_rarm.duration
        # dt = 1/120 
        # n_time_steps = int(tau_exec / dt)
        # ts = np.zeros([n_time_steps, 1])

        xs_step = np.zeros([n_time_steps, dmp_rarm.dim_x])
        xds_step = np.zeros([n_time_steps, dmp_rarm.dim_x])

        q_steps = np.zeros((n_time_steps, 4))
        q_steps[0, :] = traj_rarm.y_init[3:]
        y = np.quaternion(traj_rarm.y_init[3], traj_rarm.y_init[4], traj_rarm.y_init[5], traj_rarm.y_init[6])
        z = np.quaternion(0,0,0,0)
        x_phase, _ = dmp_rarm._phase_system_rot.integrate_start()

        x, xd = dmp_rarm.integrate_start()
        xs_step[0, :] = x
        xds_step[0, :] = xd
        for tt in range(1, n_time_steps):
            # ts[tt] = dt * tt
            dt = ts[tt] - ts[tt-1]
            xs_step[tt, :], xds_step[tt, :] = dmp_rarm.integrate_step(dt, xs_step[tt - 1, :])
            x_phase, q_step, z = dmp_rarm.integrate_step_quaternion(x_phase, y, z, dt)
            y = q_step
            q_steps[tt,:] = quaternion.as_float_array(q_step)
        traj_reproduced = dmp_rarm.states_as_trajectory(ts, xs_step, xds_step)

        cart_traj_reproduced_rarm = np.column_stack((traj_reproduced.ys, q_steps))

        for i in range(7):
            plt.figure(i+1)
            plt.plot(traj_rarm.ys[:,i], label="demonstrated")
            plt.plot(cart_traj_reproduced_ana_rarm[:,i], label="reproduced ana")
            plt.plot(cart_traj_reproduced_rarm[:,i], label="reproduced")
            plt.legend()
            plt.show()
        
        ##################### LEFT ARM
       # Integrate DMP (with same tau) 
        n_time_steps = len(traj_larm._ts)
        ts = traj_larm._ts
        # Integrate DMP (with 1.3*tau)
        # tau_exec = 1.3 * traj_larm.duration
        # dt = 1/120 
        # n_time_steps = int(tau_exec / dt)
        # ts = np.zeros([n_time_steps, 1])
        
        xs_step = np.zeros([n_time_steps, dmp_larm.dim_x])
        xds_step = np.zeros([n_time_steps, dmp_larm.dim_x])

        q_steps = np.zeros((n_time_steps, 4))
        q_steps[0, :] = traj_larm.y_init[3:]
        y = np.quaternion(traj_larm.y_init[3], traj_larm.y_init[4], traj_larm.y_init[5], traj_larm.y_init[6])
        z = np.quaternion(0,0,0,0)
        x_phase, _ = dmp_larm._phase_system_rot.integrate_start()

        x, xd = dmp_larm.integrate_start()
        xs_step[0, :] = x
        xds_step[0, :] = xd
        for tt in range(1, n_time_steps):
            # ts[tt] = dt * tt
            dt = ts[tt] - ts[tt-1]
            xs_step[tt, :], xds_step[tt, :] = dmp_larm.integrate_step(dt, xs_step[tt - 1, :])
            x_phase, q_step, z = dmp_larm.integrate_step_quaternion(x_phase, y, z, dt)
            y = q_step
            q_steps[tt,:] = quaternion.as_float_array(q_step)
        traj_reproduced = dmp_larm.states_as_trajectory(ts, xs_step, xds_step)

        cart_traj_reproduced_larm = np.column_stack((traj_reproduced.ys, q_steps))

        for i in range(7):
            plt.figure(i+1)
            plt.plot(traj_larm.ys[:,i], label="demonstrated")
            plt.plot(cart_traj_reproduced_ana_larm[:,i], label="reproduced ana")
            plt.plot(cart_traj_reproduced_larm[:,i], label="reproduced")
            plt.legend()
            plt.show()

        # NOT USED CURRENTLY
        cart_traj_reproduced = Trajectory(ts=ts, ys=cart_traj_reproduced_rarm)

        if args.show or args.save:
            ################################################
            # Plot results

            # h, axs = dmp.plot(dmp.tau,ts,xs_step,xds_step)
            # fig.canvas.set_window_title(f'Step-by-step integration (n_bfs={n_bfs})')
            # fig.savefig(Path(args.output_directory,f'dmp_trained_{n_bfs}.png'))

            h_demo, axs = traj_rarm.plot()
            h_repr, _ = cart_traj_reproduced.plot(axs)
            d = "demonstration"
            plt.setp(h_demo, linestyle="-", linewidth=4, color=(0.8, 0.8, 0.8), label=d)
            plt.setp(h_repr, linestyle="--", linewidth=2, color=(0.0, 0.0, 0.5), label="reproduced")
            plt.legend()
            plt.gcf().canvas.set_window_title(f"Comparison {d}/reproduced  (n_bfs={n_bfs})")
            plt.gcf().suptitle(f"Comparison {d}/reproduced  (n_bfs={n_bfs})")
            if args.save:
                plt.gcf().savefig(Path(args.output_directory, f"trajectory_comparison_{n_bfs}.png"))

    if args.show or args.save:
        if len(n_bfs_list) > 1:
            # Plot the mean absolute error
            ax = plt.figure().add_subplot(111)
            print(n_bfs_list)
            print(mean_absolute_errors_rarm)
            ax.plot(n_bfs_list, mean_absolute_errors_rarm)
            ax.set_xlabel("number of basis functions")
            ax.set_ylabel("mean absolute error between demonstration and reproduced")
            filename = "mean_absolute_errors.png"
            if args.save:
                plt.gcf().savefig(Path(args.output_directory, filename))

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
