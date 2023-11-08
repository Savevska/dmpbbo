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
""" Script for tuning the exploration. """


import argparse
import os
from pathlib import Path
import sys
# sys.path.append("/home/ksavevska/dmpbbo")
# sys.path.append("/home/user/talos_ws/dmpbbo")
sys.path.append("/home/hdd/ksavevska/dmpbbo")

import numpy as np
from matplotlib import pyplot as plt
from math import pi
import dmpbbo.json_for_cpp as jc
from dmpbbo.bbo.DistributionGaussian import DistributionGaussian


def main():
    """ Main function that is called when executing the script. """

    parser = argparse.ArgumentParser()
    parser.add_argument("dmp", help="input dmp")
    parser.add_argument("output_directory", help="directory to write results to")
    parser.add_argument("--sigma", help="sigma of covariance matrix", type=float, default=3.0)
    parser.add_argument("--n", help="number of samples", type=int, default=10)
    parser.add_argument("--traj", action="store_true", help="integrate DMP and save trajectory")
    parser.add_argument("--show", action="store_true", help="show result plots")
    parser.add_argument("--save", action="store_true", help="save result plots to png")
    args = parser.parse_args()

    sigma_dir = "sigma_%1.3f" % args.sigma
    directory = Path(args.output_directory, sigma_dir)

    filename = args.dmp
    print(f"Loading DMP from: {filename}")
    dmp = jc.loadjson(filename)
    ts = dmp.ts_train
    parameter_vector = dmp.get_param_vector()

    n_samples = args.n
    # sigma = args.sigma

    # Custom sigmas for each limb
    sigma_arms = np.tile(1.5, 14).reshape((1,-1))
    # sigma_arms = np.tile(1.0, 14).reshape((1,-1))

    # sigma_head = np.tile(0.1, 2).reshape((1,-1))
    # sigma_legs = np.tile(0.01, 12).reshape((1,-1))
    # sigma_torso = np.tile(0.6, 2).reshape((1,-1))

    # sigma_arms = np.array([[1.0, 1.0, 1.0, 1.0, 0.3, 0.3, 0.3, 1.0, 1.0, 1.0, 1.0, 0.3, 0.3, 0.3]])
    sigma_head = np.tile(0.01, 2).reshape((1,-1))
    # sigma_legs = 2*np.tile(0.3, 12).reshape((1,-1)) #0.01 for stable learning
    sigma_torso = np.tile(0.8, 2).reshape((1,-1))


    if "goal" in dmp._selected_param_names:
        print("goal in params")
        sigma = np.column_stack((sigma_arms, sigma_head, sigma_torso))
        sigma_goal = np.column_stack((np.tile(5*(pi/180), 14).reshape((1,-1)), np.tile(0.0, 2).reshape((1,-1)), np.tile(2*(pi/180), 2).reshape((1,-1))))
        # sigma_goal = np.column_stack((np.tile(2*(pi/180), 14).reshape((1,-1)), np.tile(0.0, 2).reshape((1,-1)), np.tile(1*(pi/180), 2).reshape((1,-1))))

        sigma = np.tile(sigma, int((dmp._dim_y*20)/sigma.size))
        sigma = np.column_stack((sigma, sigma_goal))
        # sigma = 0.2*parameter_vector
        covar_init = sigma * sigma * np.eye(parameter_vector.size)
        distribution = DistributionGaussian(parameter_vector, covar_init)
    else:
        print("goal not in params")
        # sigma = np.column_stack((sigma_arms, sigma_head, sigma_torso))
        # sigma = np.tile(sigma, int(parameter_vector.size/sigma.size))
        sigma = 0.25*parameter_vector
        covar_init = sigma * sigma * np.eye(parameter_vector.size)
        distribution = DistributionGaussian(parameter_vector, covar_init)

    filename = Path(directory, f"distribution.json")
    print(f"Saving sampling distribution to: {filename}")
    os.makedirs(directory, exist_ok=True)
    jc.savejson(filename, distribution)

    samples = distribution.generate_samples(n_samples)

    if args.show or args.save:
        fig = plt.figure()

        ax1 = fig.add_subplot(121)  # noqa
        distribution.plot(ax1)
        ax1.plot(samples[:, 0], samples[:, 1], "o", color="#BBBBBB")

        ax2 = fig.add_subplot(122)

        xs, xds, _, _ = dmp.analytical_solution()
        traj_mean = dmp.states_as_trajectory(ts, xs, xds)
        lines, _ = traj_mean.plot([ax2])
        plt.setp(lines, linewidth=4, color="#007700")

    for i_sample in range(n_samples):

        dmp.set_param_vector(samples[i_sample, :])

        filename = Path(directory, f"{i_sample:02}_dmp")
        print(f"Saving sampled DMP to: {filename}.json")
        jc.savejson(str(filename) + ".json", dmp)
        jc.savejson_for_cpp(str(filename) + "_for_cpp.json", dmp)

        if args.show or args.save or args.traj:
            xs, xds, forcing, fa_outputs = dmp.analytical_solution()
            traj_sample = dmp.states_as_trajectory(ts, xs, xds)
            if args.traj:
                filename = Path(directory, f"{i_sample:02}_traj.txt")
                print(f"Saving sampled trajectory to: {filename}")
                traj_sample.savetxt(filename)
            if args.show or args.save:
                lines, _ = traj_sample.plot([ax2])  # noqa
                plt.setp(lines, color="#BBBBBB", alpha=0.5)

    if args.save:
        filename = "exploration_dmp_traj.png"
        plt.gcf().savefig(Path(directory, filename))

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
