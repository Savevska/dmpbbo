# This file is part of DmpBbo, a set of libraries and programs for the
# black-box optimization of dynamical movement primitives.
# Copyright (C) 2018 Freek Stulp
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
"""Script for dmp demo."""

import numpy as np
import quaternion
from matplotlib import pyplot as plt
import sys
sys.path.append("/Users/kristina/WORK/dmpbbo/")
from dmpbbo.dmps.CartDmp import CartDmp
from dmpbbo.dmps.Trajectory import Trajectory
from dmpbbo.functionapproximators.FunctionApproximatorRBFN import FunctionApproximatorRBFN

from scipy.spatial.transform import Rotation
import dmpbbo.json_for_cpp as js


def main():
    """ Main function of the script. """
    tau = 0.5
    n_dims = 3
    n_time_steps = 51

    y_init = np.array([0.0, 0.7, 0.0, 1.0, 0.0, 0.0, 0.0])
    yd_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ydd_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    y_attr = np.array([0.4, 0.5, 0.0, 0.5, 0.5, 0.5, -0.5])
    yd_attr = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ydd_attr = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    ts = np.linspace(0, tau, n_time_steps)
    traj = Trajectory.from_polynomial(ts, y_init, yd_init, ydd_init, y_attr, yd_attr, ydd_attr)
    
    dmp_types = ["IJSPEERT_2002_MOVEMENT", "KULVICIUS_2012_JOINING", "COUNTDOWN_2013"]
    for dmp_type in dmp_types:
        function_apps = [FunctionApproximatorRBFN(25, 0.75) for _ in range(n_dims)]
        function_apps_rot = [FunctionApproximatorRBFN(25, 0.75) for _ in range(n_dims)]

        dmp = CartDmp.from_traj(trajectory=traj, function_approximators_pos=function_apps, function_approximators_rot=function_apps_rot, dmp_type=dmp_type)
        dmp.set_selected_param_names(["weights"])

        print(dmp.get_param_vector().shape)
        print(dmp._function_approximators[0].get_param_vector().shape)

        # dmp = CartDmp(tau, y_init, y_attr, function_apps, 20)
        print(dmp_type)
        # print(dmp._function_approximators[0]._model_params)
        # print(dmp.weights_rot)
        # print("\n")

        tau_exec = 0.5
        n_time_steps = 51
        ts = np.linspace(0, tau_exec, n_time_steps)
        # print(dmp.fa_rot.is_trained())
        # print(dmp.fa_rot._model_params)
        xs_ana, xds_ana, forcing_terms_ana, fa_outputs_ana, q_traj = dmp.analytical_solution(ts)

        # js.savejson("cdmp_test.json", dmp)
        # print(q_traj)
        # demonstrated = []
        # reproduced = []
        # for i in range(len(q_traj)):
        #     rot_demo = Rotation.from_quat(traj.ys[i, 3:]).as_euler("zyx", degrees=True)
        #     rot = Rotation.from_quat(q_traj[i, :]).as_euler("zyx", degrees=True)

        #     demonstrated.append(rot_demo)
        #     reproduced.append(rot)


        dt = ts[1]
        dim_x = xs_ana.shape[1]
        xs_step = np.zeros([n_time_steps, dim_x])
        xds_step = np.zeros([n_time_steps, dim_x])

        q_steps = np.zeros((n_time_steps, 4))
        q_steps[0, :] = y_init[3:]
        y = np.quaternion(y_init[3], y_init[4], y_init[5], y_init[6])
        z = np.quaternion(0,0,0,0)
        x_phase, _ = dmp._phase_system_rot.integrate_start()


        x, xd = dmp.integrate_start()
        xs_step[0, :] = x
        xds_step[0, :] = xd
        for tt in range(1, n_time_steps):
            xs_step[tt, :], xds_step[tt, :] = dmp.integrate_step(dt, xs_step[tt - 1, :])
            x_phase, q_step, z = dmp.integrate_step_quaternion(x_phase, y, z, dt)
            y = q_step
            q_steps[tt,:] = quaternion.as_float_array(q_step)
        
        plt.figure()
        plt.plot(traj.ys[:, 3], label="demonstrated")
        plt.plot(q_traj[:, 0], label="reproduced")
        plt.plot(q_steps[:, 0], label="steps")
        plt.legend()
        plt.show()

        # print("Plotting " + dmp_type + " DMP")

        # dmp.plot(ts, xs_ana, xds_ana, forcing_terms=forcing_terms_ana, fa_outputs=fa_outputs_ana)
        # plt.gcf().canvas.set_window_title(f"Analytical integration ({dmp_type})")

        # dmp.plot(ts, xs_step, xds_step)
        # plt.gcf().canvas.set_window_title(f"Step-by-step integration ({dmp_type})")

        # lines, axs = traj.plot()
        # plt.setp(lines, linestyle="-", linewidth=4, color=(0.8, 0.8, 0.8))
        # plt.setp(lines, label="demonstration")

    #     traj_reproduced = dmp.states_as_trajectory(ts, xs_step, xds_step)
    #     lines, axs = traj_reproduced.plot(axs)
    #     plt.setp(lines, linestyle="--", linewidth=2, color=(0.0, 0.0, 0.5))
    #     plt.setp(lines, label="reproduced")

    #     plt.legend()
    #     t = f"Comparison between demonstration and reproduced ({dmp_type})"
    #     plt.gcf().canvas.set_window_title(t)

    # plt.show()


if __name__ == "__main__":
    main()
