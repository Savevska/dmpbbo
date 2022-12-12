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
sys.path.append("/Users/kristina/WORK/dmpbbo")

from dmpbbo.dmps.Trajectory import Trajectory
import numpy as np
from matplotlib import pyplot as plt

import dmpbbo.json_for_cpp as jc
from dmpbbo.bbo.DistributionGaussian import DistributionGaussian


def main():
    """ Main function that is called when executing the script. """

    parser = argparse.ArgumentParser()
    parser.add_argument("dmp_rarm", help="input dmp")
    parser.add_argument("dmp_larm", help="input dmp")
    parser.add_argument("output_directory", help="directory to write results to")
    parser.add_argument("--sigma", help="sigma of covariance matrix", type=float, default=3.0)
    parser.add_argument("--n", help="number of samples", type=int, default=10)
    parser.add_argument("--traj", action="store_true", help="integrate DMP and save trajectory")
    parser.add_argument("--show", action="store_true", help="show result plots")
    parser.add_argument("--save", action="store_true", help="save result plots to png")
    args = parser.parse_args()

    sigma_dir = "sigma_%1.3f" % args.sigma
    directory = Path(args.output_directory, sigma_dir)

    filename_rarm = args.dmp_rarm
    print(f"Loading DMP from: {filename_rarm}")
    dmp_rarm = jc.loadjson(filename_rarm)
    ts_rarm = dmp_rarm.ts_train
    parameter_vector_rarm = dmp_rarm.get_param_vector()

    filename_larm = args.dmp_larm
    print(f"Loading DMP from: {filename_larm}")
    dmp_larm = jc.loadjson(filename_larm)
    ts_larm = dmp_larm.ts_train
    parameter_vector_larm = dmp_larm.get_param_vector()
    
    n_samples = args.n
    # sigma = args.sigma

    # Custom sigmas for position and orientation
    sigma_pos_rarm = np.tile(1.0, 3).reshape((1,-1))
    sigma_rot_rarm = np.tile(1.0, 3).reshape((1,-1))
    sigma_rarm = np.column_stack((sigma_pos_rarm, sigma_rot_rarm))
    sigma_rarm = np.tile(sigma_rarm, int(parameter_vector_rarm.size/sigma_rarm.size))
    covar_init_rarm = sigma_rarm * sigma_rarm * np.eye(parameter_vector_rarm.size)
    distribution_rarm = DistributionGaussian(parameter_vector_rarm, covar_init_rarm)

    sigma_pos_larm = np.tile(1.0, 3).reshape((1,-1))
    sigma_rot_larm = np.tile(1.0, 3).reshape((1,-1))
    sigma_larm = np.column_stack((sigma_pos_larm, sigma_rot_larm))
    sigma_larm = np.tile(sigma_larm, int(parameter_vector_larm.size/sigma_larm.size))
    covar_init_larm = sigma_larm * sigma_larm * np.eye(parameter_vector_larm.size)
    distribution_larm = DistributionGaussian(parameter_vector_larm, covar_init_larm)
    # distribution_rot = []
    # for i in range(parameter_vector_rot.shape[0]):
    #     print(sigma_rot.shape)
    #     sigma_rot = np.tile(sigma_rot, int(parameter_vector_rot.shape[1]/sigma_rot.shape[1]))
    #     covar_init_rot = sigma_rot * sigma_rot * np.eye(parameter_vector_rot.shape[1])
    #     dist_rot = DistributionGaussian(parameter_vector_rot[i,:], covar_init_rot)
    #     # distribution_rot.append(dist_rot)
    #     samples_rot = dist_rot.generate_samples(n_samples)

    filename_rarm = Path(directory, f"distribution_rarm.json")
    filename_larm = Path(directory, f"distribution_larm.json")

    print(f"Saving sampling distribution to: {filename_rarm}")
    os.makedirs(directory, exist_ok=True)
    jc.savejson(filename_rarm, distribution_rarm)
    print(f"Saving sampling distribution to: {filename_larm}")
    os.makedirs(directory, exist_ok=True)
    jc.savejson(filename_larm, distribution_larm)
    
    samples_rarm = distribution_rarm.generate_samples(n_samples)
    samples_larm = distribution_larm.generate_samples(n_samples)
        
    # NOT USED CURRENTLY (made only for the right arm but the parameters show and save are not used at the moment)
    if args.show or args.save:
        fig = plt.figure()

        ax1 = fig.add_subplot(121)  # noqa
        distribution_rarm.plot(ax1)
        ax1.plot(samples_rarm[:, 0], samples_rarm[:, 1], "o", color="#BBBBBB")

        ax2 = fig.add_subplot(122)

        xs, xds, _, _, q_traj = dmp_rarm.analytical_solution()
        traj_mean = dmp_rarm.states_as_trajectory(ts_rarm, xs, xds)
        traj_mean = Trajectory(ts=traj_mean.ts, ys=np.column_stack((traj_mean.ys, q_traj)))

        lines, _ = traj_mean.plot([ax2])
        plt.setp(lines, linewidth=4, color="#007700")

    for i_sample in range(n_samples):

        dmp_rarm.set_param_vector(samples_rarm[i_sample, :])
        dmp_larm.set_param_vector(samples_larm[i_sample, :])

        
        filename_rarm = Path(directory, f"{i_sample:02}_dmp_rarm")
        print(f"Saving sampled DMP to: {filename_rarm}.json")
        jc.savejson(str(filename_rarm) + ".json", dmp_rarm)

        filename_larm = Path(directory, f"{i_sample:02}_dmp_larm")
        print(f"Saving sampled DMP to: {filename_larm}.json")
        jc.savejson(str(filename_larm) + ".json", dmp_larm)
        # jc.savejson_for_cpp(str(filename) + "_for_cpp.json", dmp)

        # NOT USED CURRENTLY (made only for the right arm but the parameters show and save are not used at the moment)
        if args.show or args.save or args.traj:
            xs, xds, forcing, fa_outputs, q_traj = dmp_rarm.analytical_solution()
            traj_sample = dmp_rarm.states_as_trajectory(ts_rarm, xs, xds)
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
