# This file is part of DmpBbo, a set of libraries and programs for the
# black-box optimization of dynamical movement primitives.
# Copyright (C) 2014 Freek Stulp, ENSTA-ParisTech
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
""" Script for defining the task. """

import argparse
import os
from pathlib import Path
import sys
sys.path.append("/home/ksavevska/dmpbbo")
# sys.path.append("/Users/kristina/WORK/dmpbbo")
import jsonpickle
import dmpbbo.json_for_cpp as jc

from TaskReach import TaskReach
from dmpbbo.dmps.Trajectory import Trajectory


def main():
    """ Main function that is called when executing the script. """

    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="directory to write task to")
    parser.add_argument("filename", help="file to write task to")
    parser.add_argument("traj_filename", help="file to read trajectory from")

    args = parser.parse_args()

    ee_pos_goal = [0.65, -0.4, 0.0] 
    pos_margin = 0
    ref_cop = [0.0, 0.0]
    # stability_weight = 0.1
    # goal_weight = 1
    # traj_weight = 0 

    # stability_weight = 0.00005
    # goal_weight = 1.3
    # goal_orientation_weight = 0.7
    # traj_weight = 0.0 

    stability_weight = 5.0
    goal_weight = 4.0
    goal_orientation_weight = 0.5
    acc_weight = 0.5
    traj_weight = 0.5 

    dmp_0 = jc.loadjson(args.traj_filename)
    ts_rarm = dmp_0.ts_train
    xs, xds, forcing, fa_outputs, q_traj = dmp_0.analytical_solution()        
    traj_demonstrated = dmp_0.states_as_trajectory(ts_rarm, xs, xds)   
    # traj_demonstrated = None #Trajectory.loadtxt(args.traj_filename, 0)
    task = TaskReach(ee_pos_goal, pos_margin, ref_cop, stability_weight, goal_weight, goal_orientation_weight, acc_weight, traj_weight, traj_demonstrated)

    # Save the task instance itself
    os.makedirs(args.directory, exist_ok=True)
    filename = Path(args.directory, args.filename)
    print(f"  * Saving task to file {filename}")
    json = jsonpickle.encode(task)
    with open(filename, "w") as text_file:
        text_file.write(json)


if __name__ == "__main__":
    main()
