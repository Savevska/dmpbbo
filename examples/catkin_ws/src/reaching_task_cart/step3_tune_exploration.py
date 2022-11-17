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
sys.path.append("/home/ksavevska/dmpbbo")

from dmpbbo.dmps.Trajectory import Trajectory
import numpy as np
from matplotlib import pyplot as plt

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
    parameter_vector_rot = dmp.weights_rot # ? is it the weights?
    
    n_samples = args.n
    # sigma = args.sigma

    # Custom sigmas for position and orientation
    sigma_pos = np.tile(1.0, 3).reshape((1,-1))
    sigma_rot = np.tile(1.0, 3).reshape((1,-1))

    # sigma = np.column_stack((sigma_pos, sigma_rot))
    sigma_pos = np.tile(sigma_pos, int(parameter_vector.size/sigma_pos.size))
    covar_init = sigma_pos * sigma_pos * np.eye(parameter_vector.size)
    distribution = DistributionGaussian(parameter_vector, covar_init)

    distribution_rot = []
    for i in range(parameter_vector_rot.shape[0]):
        print(sigma_rot.shape)
        sigma_rot = np.tile(sigma_rot, int(parameter_vector_rot.shape[1]/sigma_rot.shape[1]))
        covar_init_rot = sigma_rot * sigma_rot * np.eye(parameter_vector_rot.shape[1])
        dist_rot = DistributionGaussian(parameter_vector_rot[i,:], covar_init_rot)
        # distribution_rot.append(dist_rot)
        samples_rot = dist_rot.generate_samples(n_samples)

    filename = Path(directory, f"distribution.json")
    filename_rot = Path(directory, f"distribution_rot.json")

    print(f"Saving sampling distribution to: {filename}")
    os.makedirs(directory, exist_ok=True)
    jc.savejson(filename, distribution)
    os.makedirs(directory, exist_ok=True)
    jc.savejson(filename_rot, distribution_rot)

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
        dmp.weights_rot = samples_rot[i_sample, :]
        
        filename = Path(directory, f"{i_sample:02}_dmp")
        print(f"Saving sampled DMP to: {filename}.json")
        jc.savejson(str(filename) + ".json", dmp)
        # jc.savejson_for_cpp(str(filename) + "_for_cpp.json", dmp)

        if args.show or args.save or args.traj:
            xs, xds, forcing, fa_outputs, q_traj = dmp.analytical_solution()
            traj_sample = dmp.states_as_trajectory(ts, xs, xds)
            trajectory = Trajectory(ts=traj_sample.ts, ys=np.column_stack((traj_sample.ys, q_traj)))

            if args.traj:
                filename = Path(directory, f"{i_sample:02}_traj.txt")
                print(f"Saving sampled trajectory to: {filename}")
                trajectory.savetxt(filename)
            if args.show or args.save:
                lines, _ = trajectory.plot([ax2])  # noqa
                plt.setp(lines, color="#BBBBBB", alpha=0.5)

    if args.save:
        filename = "exploration_dmp_traj.png"
        plt.gcf().savefig(Path(directory, filename))

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
