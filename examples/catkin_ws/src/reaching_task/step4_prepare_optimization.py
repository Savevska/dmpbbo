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
""" Script for preparing the optimization. """
import sys
# sys.path.append("/home/ksavevska/dmpbbo")
sys.path.append("/home/user/talos_ws/dmpbbo")
import argparse
from pathlib import Path

import jsonpickle

from dmpbbo.bbo.updaters import UpdaterCovarAdaptation, UpdaterCovarDecay, UpdaterMean
from dmpbbo.bbo_of_dmps.step_by_step_optimization import prepare_optimization


def main():
    """ Main function that is called when executing the script. """

    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="directory to write results to")
    parser.add_argument("--traj", action="store_true", help="integrate DMP and save trajectory")
    args = parser.parse_args()

    filename = Path(args.directory, "task.json")
    with open(filename, "r") as f:
        task = jsonpickle.decode(f.read())

    filename = Path(args.directory, "distribution_initial.json")
    with open(filename, "r") as f:
        distribution_init = jsonpickle.decode(f.read())

    filename = Path(args.directory, "dmp_initial.json")
    with open(filename, "r") as f:
        dmp = jsonpickle.decode(f.read())

    n_samples_per_update = 30

    updater_name = "adapt"
    if updater_name == "mean":
        updater = UpdaterMean(eliteness=10, weighting="PI-BB")
    elif updater_name == "decay":
        updater = UpdaterCovarDecay(eliteness=10, weighting="PI-BB", covar_decay_factor=0.99)
    else:
        updater = UpdaterCovarAdaptation(
            eliteness=10,
            weighting="PI-BB",
            diagonal_max=None,
            diagonal_min=None,
            diag_only=False,
            learning_rate=0.3,
        )

    task_solver = None
    prepare_optimization(
        args.directory,
        task,
        task_solver,
        distribution_init,
        n_samples_per_update,
        updater,
        dmp,
        args.traj,
    )


if __name__ == "__main__":
    main()
