/**
 * @file   MetaParametersRBFN.hpp
 * @brief  MetaParametersRBFN class header file.
 * @author Freek Stulp
 *
 * This file is part of DmpBbo, a set of libraries and programs for the 
 * black-box optimization of dynamical movement primitives.
 * Copyright (C) 2014 Freek Stulp, ENSTA-ParisTech
 * 
 * DmpBbo is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 * 
 * DmpBbo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with DmpBbo.  If not, see <http://www.gnu.org/licenses/>.
 */
 
#ifndef METAPARAMETERSRBFN_H
#define METAPARAMETERSRBFN_H

#include "functionapproximators/MetaParameters.hpp"


#include <iosfwd>
#include <vector>
#include <eigen3/Eigen/Core>

#include <nlohmann/json_fwd.hpp>

namespace DmpBbo {

/** \brief Meta-parameters for the Radial Basis Function Network (RBFN) function approximator
 * \ingroup FunctionApproximators
 * \ingroup RBFN
 */
class MetaParametersRBFN : public MetaParameters
{
  
public:
  
  /** Constructor for the algorithmic meta-parameters of the RBFN function approximator.
   *  \param[in] expected_input_dim         The dimensionality of the data this function approximator expects. Although this information is already contained in the 'centers_per_dim' argument, we ask the user to pass it explicitly so that various checks on the arguments may be conducted.
   *  \param[in] centers_per_dim Centers of the basis functions, one VectorXd for each dimension.
   *  \param[in] intersection_height The value at which two neighbouring basis functions will intersect.
   * \param[in] regularization Regularization parameter
   */
   MetaParametersRBFN(int expected_input_dim, const std::vector<Eigen::VectorXd>& centers_per_dim, double intersection_height=0.5, double regularization=0.0);
		 
  /** Constructor for the algorithmic meta-parameters of the RBFN function approximator.
   *  \param[in] expected_input_dim         The dimensionality of the data this function approximator expects. Although this information is already contained in the 'centers' argument, we ask the user to pass it explicitly so that various checks on the arguments may be conducted.
   *  \param[in] n_basis_functions_per_dim  Number of basis functions
   *  \param[in] intersection_height The value at which two neighbouring basis functions will intersect.
   * \param[in] regularization Regularization parameter
   *
   *  The centers and widths of the basis functions are determined from these parameters once the
   *  range of the input data is known, see also setInputMinMax()
   */
  MetaParametersRBFN(int expected_input_dim, const Eigen::VectorXi& n_basis_functions_per_dim, double intersection_height=0.5, double regularization=0.0);
  
  
  /** Constructor for the algorithmic meta-parameters of the RBFN function approximator.
   * This is for the special case when the dimensionality of the input data is 1.
   *  \param[in] expected_input_dim         The dimensionality of the data this function approximator expects. Since this constructor is for 1-D input data only, we simply check if this argument is equal to 1.
   *  \param[in] n_basis_functions  Number of basis functions for the one dimension
   *  \param[in] intersection_height The value at which two neighbouring basis functions will intersect.
   * \param[in] regularization Regularization parameter
   *
   *  The centers and widths of the basis functions are determined from these parameters once the
   *  range of the input data is known, see also setInputMinMax()
   */
	MetaParametersRBFN(int expected_input_dim, int n_basis_functions=10, double intersection_height=0.5, double regularization=0.0);

	static MetaParametersRBFN* from_jsonpickle(nlohmann::json json);
  
	/** Accessor function for regularization.
	 * \return Regularization parameter.
	 */
  double regularization(void) const
  {
    return regularization_;
  }
  
	/** Get the centers and widths of the basis functions.
	 *  \param[in] min Minimum values of input data (one value for each dimension).
	 *  \param[in] max Maximum values of input data (one value for each dimension).
	 *  \param[out] centers Centers of the basis functions (matrix of size n_basis_functions X n_input_dims
	 *  \param[out] widths Widths of the basis functions (matrix of size n_basis_functions X n_input_dims
	 *
	 * The reason why there are not two functions getCenters and getWidths is that it is much easier
	 * to compute both at the same time, and usually you will need both at the same time anyway.
	 */
	void getCentersAndWidths(const Eigen::VectorXd& min, const Eigen::VectorXd& max, Eigen::MatrixXd& centers, Eigen::MatrixXd& widths) const;
	
	MetaParametersRBFN* clone(void) const;

	std::string toString(void) const;

  // https://github.com/nlohmann/json/issues/1324
  friend void to_json(nlohmann::json& j, const MetaParametersRBFN& m);
  //friend void from_json(const nlohmann::json& j, MetaParametersRBFN& m);
  
private:
  Eigen::VectorXi n_bfs_per_dim_; // should be const
  std::vector<Eigen::VectorXd> centers_per_dim_; // should be const
  double intersection_height_; // should be const
  double regularization_; // should be const

  /**
   * Default constructor.
   * \remarks This default constuctor is required for boost::serialization to work. Since this
   * constructor should not be called by other classes, it is private (boost::serialization is a
   * friend)
   */
  MetaParametersRBFN(void) {}; 

};

}

#endif        //  #ifndef METAPARAMETERSRBFN_H

