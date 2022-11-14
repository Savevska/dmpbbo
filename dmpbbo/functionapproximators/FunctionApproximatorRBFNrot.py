# This file is part of DmpBbo, a set of libraries and programs for the
# black-box optimization of dynamical movement primitives.
# Copyright (C) 2018, 2022 Freek Stulp
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
""" Module for the FunctionApproximatorRBFN class. """


import numpy as np
from matplotlib import pyplot as plt

from dmpbbo.functionapproximators.basis_functions import Gaussian
from dmpbbo.functionapproximators.FunctionApproximator import FunctionApproximator
from dmpbbo.functionapproximators.FunctionApproximatorWLS import FunctionApproximatorWLS


class FunctionApproximatorRBFNrot(FunctionApproximator):
    """ A radial basis function network (RBFN)  function approximator.
    """

    def __init__(self, n_bfs_per_dim, intersection_height=0.7, regularization=0.0):
        """Initialize an RBNF function approximator.

        @param n_bfs_per_dim: Number of basis functions per input dimension.
        @param intersection_height: Relative value at which two neighbouring basis functions
            will intersect (default=0.7)
        @param regularization: Regularization parameter (default=0.0)
        """
        meta_params = {
            "n_basis_functions_per_dim": np.atleast_1d(n_bfs_per_dim),
            "intersection_height": intersection_height,
            "regularization": regularization,
        }

        model_param_names = ["centers", "widths", "weights"]

        super().__init__(meta_params, model_param_names)
    
    @staticmethod
    def get_centers_and_widths(inputs, n_bfs_per_dim, intersection_height=0.7):
        """Get the centers and widths of basis functions.

        @param inputs: The input data (size: n_samples X n_dims)
        @param n_bfs_per_dim: Number of basis functions per input dimension.
        @param intersection_height: The relative value at which two neighbouring basis functions
            will intersect (default=0.7)
        @return: centers: Centers of the basis functions (n_basis_functions X n_input_dims)
            widths: Widths of the basis functions (n_basis_functions X n_input_dims)
        """
        min_vals = inputs.min(axis=0)
        max_vals = inputs.max(axis=0)
        min_vals = np.atleast_1d(min_vals)
        max_vals = np.atleast_1d(max_vals)
        n_dims = len(min_vals)
        n_bfs_per_dim = np.atleast_1d(n_bfs_per_dim)
        if n_bfs_per_dim.size < n_dims:
            if n_bfs_per_dim.size == 1:
                n_bfs_per_dim = n_bfs_per_dim * np.ones(n_dims).astype(int)
            else:
                raise ValueError(f"n_bfs_per_dim should be of size {n_dims}")

        centers_per_dim_local = []
        widths_per_dim_local = []
        for i_dim in range(n_dims):
            n_bfs = n_bfs_per_dim[i_dim]

            cur_centers = np.linspace(min_vals[i_dim], max_vals[i_dim], n_bfs)
            # cur_centers = np.exp(-2.0*np.linspace(min_vals[i_dim], max_vals[i_dim], n_bfs))
            # Determine the widths from the centers
            cur_widths = np.ones(n_bfs)
            h = intersection_height
            if n_bfs > 1:
                # Consider 2 neighbouring functions, exp(-0.5(x-c0)^2/w^2) and exp(-0.5(x-c1)^2/w^2)
                # Assuming same widths, they are certain to intersect at x = 0.5(c0+c1)
                # And we want the activation at x to be 'intersection'. So
                #            y = exp(-0.5(x-c0)^2/w^2)
                # intersection = exp(-0.5((0.5(c0+c1))-c0)^2/w^2)
                # intersection = exp(-0.5((0.5*c1-0.5*c0)^2/w^2))
                # intersection = exp(-0.5((0.5*(c1-c0))^2/w^2))
                # intersection = exp(-0.5(0.25*(c1-c0)^2/w^2))
                # intersection = exp(-0.125((c1-c0)^2/w^2))
                #            w = sqrt((c1-c0)^2/-8*ln(intersection))
                for cc in range(n_bfs - 1):
                    w = np.sqrt(np.square(cur_centers[cc + 1] - cur_centers[cc]) / (-8 * np.log(h)))
                    cur_widths[cc] = w

                cur_widths[n_bfs - 1] = cur_widths[n_bfs - 2]
                # cur_widths = np.square(np.diff(cur_centers)*h)
                # cur_widths = np.append(cur_widths, cur_widths[-1])
            centers_per_dim_local.append(cur_centers)
            widths_per_dim_local.append(cur_widths)

        # We now have the centers and widths for each dimension separately.
        # This is like meshgrid.flatten, but then for any number of dimensions
        # I'm sure numpy has better functions for this, but I could not find them, and I already
        # had the code in C++.
        digit_max = n_bfs_per_dim
        n_centers = np.prod(digit_max)
        digit = [0] * n_dims

        centers = np.zeros((n_centers, n_dims))
        widths = np.zeros((n_centers, n_dims))
        i_center = 0
        while digit[0] < digit_max[0]:
            for i_dim in range(n_dims):
                centers[i_center, i_dim] = centers_per_dim_local[i_dim][digit[i_dim]]
                widths[i_center, i_dim] = widths_per_dim_local[i_dim][digit[i_dim]]
            i_center += 1

            # Increment last digit by one
            digit[n_dims - 1] += 1
            for i_dim in range(n_dims - 1, 0, -1):
                if digit[i_dim] >= digit_max[i_dim]:
                    digit[i_dim] = 0
                    digit[i_dim - 1] += 1
        # print(centers)
        # print(widths)
        return centers, widths


    @staticmethod
    def _train(inputs, targets, meta_params, **kwargs):
        # Determine the centers and widths of the basis functions, given the input data range
        n_bfs_per_dim = meta_params["n_basis_functions_per_dim"]
        n_bfs = np.prod(n_bfs_per_dim)

        height = meta_params["intersection_height"]
        centers, widths = FunctionApproximatorRBFNrot.get_centers_and_widths(inputs, n_bfs_per_dim, height)
        model_params = {"centers": centers, "widths": widths}

        # Get the activations of the basis functions
        activations = FunctionApproximatorRBFNrot._activations(inputs, model_params)

        # Prepare the least squares function approximator and train it
        use_offset = False
        regularization = meta_params["regularization"]

        # n_samples = targets.size
        n_samples = targets.shape[0]
        # inputs = inputs.reshape(n_samples, -1)

        # Make the design matrix
        if use_offset:
            # Add a column with 1s
            X = np.column_stack((activations, np.ones(n_samples)))  # noqa
        else:
            X = activations  # noqa

        # Weights matrix
        weights = kwargs.get("weights", None)
        if weights is None:
            W = np.eye(n_samples)  # noqa
        else:
            W = np.diagflat(weights)  # noqa

        # Regularization matrix
        n_dims_X = X.shape[1]  # noqa
        Gamma = regularization * np.identity(n_dims_X)  # noqa

        # Compute beta
        # 1 x n_betas
        # = inv( (n_betas x n_sam)*(n_sam x n_sam)*(n_sam*n_betas) )*
        #                                          ( (n_betas x n_sam)*(n_sam x n_sam)*(n_sam * 1) )
        # = inv(n_betas x n_betas)*(n_betas x 1)
        #
        # Least squares is a one-liner
        # Apparently, not everybody has python3.5 installed, so don't use @
        # betas = np.linalg.inv(X.T@W@X + Gamma)@X.T@W@targets
        # In python<=3.4, it is not a one-liner
        to_invert = np.dot(np.dot(X.T, W), X) + Gamma
        beta = np.dot(np.dot(np.dot(np.linalg.inv(to_invert), X.T), W), targets)
        # A = np.multiply(inputs, np.divide(activations, np.sum(activations)))
        # beta = np.transpose(np.linalg.lstsq(A, targets)[0])
        # print(beta)
        if use_offset:
            model_params_wls = {"slope": beta[:-1], "offset": beta[-1]}
        else:
            model_params_wls = {"slope": beta}

        model_params["weights"] = model_params_wls["slope"].reshape(n_bfs, -1)

        return model_params

    @staticmethod
    def _activations(inputs, model_params):
        """Get the activations for given centers, widths and inputs.

        @param centers: The center of the basis function (size: n_basis_functions X n_dims)
        @param widths: The width of the basis function (size: n_basis_functions X n_dims)
        @param inputs: The input data (size: n_samples X n_dims)
        @param normalized_basis_functions: Whether to normalize the basis functions (default=False)

        @return: The kernel activations, computed for each of the samples in the input data
        (size: n_samples X n_basis_functions)
        """
        centers = model_params["centers"]
        widths = model_params["widths"]
        normalized_basis_functions = True

        print(centers)
        print(widths)

        n_samples = inputs.shape[0]
        n_basis_functions = centers.shape[0]
        n_dims = centers.shape[1] if len(centers.shape) > 1 else 1
        if n_dims == 1:
            # Make sure arguments have shape (N,1) not (N,)
            centers = centers.reshape(n_basis_functions, 1)
            widths = widths.reshape(n_basis_functions, 1)
            inputs = inputs.reshape(n_samples, 1)

        kernel_activations = np.ones([n_samples, n_basis_functions])

        if normalized_basis_functions and n_basis_functions == 1:
            # Normalizing one Gaussian basis function with itself leads to 1 everywhere.
            kernel_activations.fill(1.0)
            return kernel_activations

        for bb in range(n_basis_functions):
            # Here, we compute the values of a (unnormalized) multi-variate Gaussian:
            #   activation = exp(-0.5*(x-mu)*Sigma^-1*(x-mu))
            # Because Sigma is diagonal in our case, this simplifies to
            #   activation = exp(\sum_d=1^D [-0.5*(x_d-mu_d)^2/Sigma_(d,d)])
            #              = \prod_d=1^D exp(-0.5*(x_d-mu_d)^2/Sigma_(d,d))
            # This last product is what we compute below incrementally

            for i_dim in range(n_dims):
                c = centers[bb, i_dim]
                w = widths[bb, i_dim]
                for i_s in range(n_samples):
                    x = inputs[i_s, i_dim]
                    kernel_activations[i_s, bb] *= np.exp(-0.5 * np.square(x - c) / (w * w))

        if normalized_basis_functions:
            # Normalize the basis value; they should sum to 1.0 for each time step.
            for i_sample in range(n_samples):
                sum_kernel_activations = kernel_activations[i_sample, :].sum()
                for i_basis in range(n_basis_functions):
                    if sum_kernel_activations == 0.0:
                        # Apparently, no basis function was active. Set all to same value
                        kernel_activations[i_sample, i_basis] = 1.0 / n_basis_functions
                    else:
                        # Standard case, normalize so that they sum to 1.0
                        kernel_activations[i_sample, i_basis] /= sum_kernel_activations

        return kernel_activations

    @staticmethod
    def _predict(inputs, model_params):
        acts = FunctionApproximatorRBFNrot._activations(inputs, model_params)
        weighted_acts = np.zeros(acts.shape)
        for ii in range(acts.shape[1]):
            weighted_acts[:, ii] = acts[:, ii] * model_params["weights"][ii]
        return weighted_acts.sum(axis=1)

    def plot_model_parameters(self, inputs_min, inputs_max, **kwargs):
        """ Plot a representation of the model parameters on a grid.

        @param inputs_min: The min values for the grid
        @param inputs_max:  The max values for the grid
        @return: line handles and axis
        """
        inputs, n_samples_per_dim = FunctionApproximator._get_grid(inputs_min, inputs_max)
        activations = self._activations(inputs, self._model_params)

        ax = kwargs.get("ax") or self._get_axis()

        lines = self._plot_grid_values(inputs, activations, ax, n_samples_per_dim)
        alpha = 1.0 if self.dim_input() < 2 else 0.3
        plt.setp(lines, color=[0.7, 0.7, 0.7], linewidth=1, alpha=alpha)

        return lines, ax
