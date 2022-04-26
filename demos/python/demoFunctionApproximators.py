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
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DmpBbo.  If not, see <http://www.gnu.org/licenses/>.


import matplotlib.pyplot as plt

from dmpbbo.functionapproximators.FunctionApproximatorLWR import *
from dmpbbo.functionapproximators.FunctionApproximatorRBFN import *


def targetFunction(n_samples_per_dim):

    n_dims = 1 if np.isscalar(n_samples_per_dim) else len(n_samples_per_dim)

    if n_dims == 1:
        inputs = np.linspace(0.0, 2.0, n_samples_per_dim)
        targets = 3 * np.exp(-inputs) * np.sin(2 * np.square(inputs))

    else:
        n_samples = np.prod(n_samples_per_dim)
        # Here comes naive inefficient implementation...
        x1s = np.linspace(-2.0, 2.0, n_samples_per_dim[0])
        x2s = np.linspace(-2.0, 2.0, n_samples_per_dim[1])
        inputs = np.zeros((n_samples, n_dims))
        targets = np.zeros(n_samples)
        ii = 0
        for x1 in x1s:
            for x2 in x2s:
                inputs[ii, 0] = x1
                inputs[ii, 1] = x2
                targets[ii] = 2.5 * x1 * np.exp(-np.square(x1) - np.square(x2))
                ii += 1

    return inputs, targets


def train(fa_name, n_dims):

    # Generate training data
    n_samples_per_dim = 30 if n_dims == 1 else [10, 10]
    (inputs, targets) = targetFunction(n_samples_per_dim)

    n_rfs = 9 if n_dims == 1 else [5, 5]  # Number of basis functions. To be used later.

    # Initialize function approximator
    if fa_name == "WLS":
        use_offset = True
        regularization = 0.1
        fa = FunctionApproximatorWLS(use_offset, regularization)
    elif fa_name == "LWR":
        # This value for intersection is quite low. But for the demo it is nice
        # because it makes the linear segments quite obvious.
        intersection = 0.2
        fa = FunctionApproximatorLWR(n_rfs, intersection)
    else:
        intersection = 0.7
        fa = FunctionApproximatorRBFN(n_rfs, intersection)

    # Train function approximator with data
    fa.train(inputs, targets)

    # Make predictions for the targets
    outputs = fa.predict(inputs)

    # Plotting
    (h, ax) = fa.plot(
        inputs, targets=targets, plot_residuals=True, plot_basis_functions=True
    )
    ax.set_title(fa_name + " " + str(n_dims) + "D")
    plt.gcf().canvas.set_window_title(fa_name + " " + str(n_dims) + "D")


if __name__ == "__main__":
    """Run some training sessions and plot results."""

    for fa_name in ["WLS", "RBFN", "LWR"]:
        for n_dims in [1, 2]:
            train(fa_name, n_dims)

    plt.show()
