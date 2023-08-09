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
import numpy as np
from matplotlib import pyplot as plt

import dmpbbo.json_for_cpp as jc
from dmpbbo.dmps.Dmp import Dmp
from dmpbbo.dmps.Trajectory import Trajectory
from dmpbbo.functionapproximators.FunctionApproximatorRBFN import FunctionApproximatorRBFN
from dmpbbo.dynamicalsystems.SigmoidSystem import SigmoidSystem


def main():
    
    """ Main function that is called when executing the script. """

    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory_file", help="file to read trajectory from")
    parser.add_argument("output_directory", help="directory to write dmp and other results to")
    parser.add_argument("--n", help="max number of basis functions", type=int, default=15)
    parser.add_argument("--show", action="store_true", help="Show plots")
    parser.add_argument("--save", action="store_true", help="save result plots to png")
    args = parser.parse_args()

    os.makedirs(args.output_directory, exist_ok=True)

    ################################################
    # Read trajectory and train DMP with it.

    print(f"Reading trajectory from: {args.trajectory_file}\n")
    traj = Trajectory.loadtxt(args.trajectory_file, 0)
    filename_traj = Path(args.output_directory, "trajectory.txt")
    traj.savetxt(filename_traj)
    # jc.savejson(traj,Path(args.output_directory,'trajectory.json'))
    n_dims = traj.dim
    traj._misc = None
    print(traj._misc)
    peak_to_peak = np.ptp(traj.ys, axis=0)  # Range of data; used later on
    # right_limb_final = [0.45102077060577095, -1.2677962295220508, 0.08608905862672511, -0.7722077471044404, 0.00823741156637503, -0.05063002474194711, 0.1065213803589149]
    # left_limb_final = [0.557265918511983, 0.5148896301504173, -0.8576324295574826, -0.4173278946408434, 0.009230304758586882, 0.009545917116773772, -0.024562330306767244]
    # right_limb_final = [0.4899644814835762, -0.7511597260092717, -1.02332355451394, -1.0503801616230817, -0.5821479894894308, 0.21006275786963968, -0.2403049932018586]
    # head_final = [0.00584903160023309, -5.8102036399887425e-06]
    # torso_final = [0.4051831181022516, 0.46660193412117756]
    # traj.ys[-1, 0:7]  = left_limb_final
    # traj.ys[-1, 7:14] = right_limb_final
    # traj.ys[-1, 14:16] = head_final
    # traj.ys[-1, 16:18] = torso_final

    mean_absolute_errors = []
    n_bfs_list = list(range(20, args.n + 1))
    for n_bfs in n_bfs_list:

        function_apps = [FunctionApproximatorRBFN(n_bfs, 0.7) for _ in range(n_dims)]
        dmp = Dmp.from_traj(traj, function_apps, dmp_type="KULVICIUS_2012_JOINING", gating_system=SigmoidSystem(tau=traj.duration, x_init=1, max_rate=-1.0, inflection_ratio=0.9))

        # These are the parameters that will be optimized.
        # dmp.set_selected_param_names(["weights"])
        dmp.set_selected_param_names(["weights", "goal"])


        ################################################
        # Save DMP to file
        d = args.output_directory
        filename = Path(d, f"dmp_trained_{n_bfs}.json")
        print(f"Saving trained DMP to: {filename}")
        jc.savejson(filename, dmp)
        jc.savejson_for_cpp(Path(d, f"dmp_trained_{n_bfs}_for_cpp.json"), dmp)

        ################################################
        # Analytical solution to compute difference

        ts = traj.ts
        xs_ana, xds_ana, _, _ = dmp.analytical_solution(ts)
        traj_reproduced_ana = dmp.states_as_trajectory(ts, xs_ana, xds_ana)

        mae = np.mean(abs(traj.ys - traj_reproduced_ana.ys))
        mean_absolute_errors.append(mae)
        print()
        print(f"               Number of basis functions: {n_bfs}")
        print(f"MAE between demonstration and reproduced: {mae}")
        print(f"                           Range of data: {peak_to_peak}")
        print()

        ################################################
        # Integrate DMP (with 1.3*tau)

        # tau_exec = 1.3 * traj.duration
        # dt = 1/120 
        # n_time_steps = int(tau_exec / dt)
        # ts = np.zeros([n_time_steps, 1])
        # xs_step = np.zeros([n_time_steps, dmp.dim_x])
        # xds_step = np.zeros([n_time_steps, dmp.dim_x])

        # x, xd = dmp.integrate_start()
        # xs_step[0, :] = x
        # xds_step[0, :] = xd
        # for tt in range(1, n_time_steps):
        #     ts[tt] = dt * tt
        #     xs_step[tt, :], xds_step[tt, :] = dmp.integrate_step(dt, xs_step[tt - 1, :])

        # traj_reproduced = dmp.states_as_trajectory(ts, xs_step, xds_step)

        # Integrate DMP (with same tau)
        
        n_time_steps = len(traj._ts)
        ts = traj._ts
        xs_step = np.zeros([n_time_steps, dmp.dim_x])
        xds_step = np.zeros([n_time_steps, dmp.dim_x])

        x, xd = dmp.integrate_start()
        xs_step[0, :] = x
        xds_step[0, :] = xd
        for tt in range(1, n_time_steps):
            dt = ts[tt] - ts[tt-1]
            xs_step[tt, :], xds_step[tt, :] = dmp.integrate_step(dt, xs_step[tt - 1, :])

        traj_reproduced = dmp.states_as_trajectory(ts, xs_step, xds_step)

        if args.show or args.save:
            ################################################
            # Plot results

            # h, axs = dmp.plot(dmp.tau,ts,xs_step,xds_step)
            # fig.canvas.set_window_title(f'Step-by-step integration (n_bfs={n_bfs})')
            # fig.savefig(Path(args.output_directory,f'dmp_trained_{n_bfs}.png'))

            h_demo, axs = traj.plot()
            h_repr, _ = traj_reproduced.plot(axs)
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
            print(mean_absolute_errors)
            ax.plot(n_bfs_list, mean_absolute_errors)
            ax.set_xlabel("number of basis functions")
            ax.set_ylabel("mean absolute error between demonstration and reproduced")
            filename = "mean_absolute_errors.png"
            if args.save:
                plt.gcf().savefig(Path(args.output_directory, filename))

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
