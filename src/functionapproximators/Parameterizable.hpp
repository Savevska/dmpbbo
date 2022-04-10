/**
 * @file   Parameterizable.hpp
 * @brief  Parameterizable class header file.
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

#ifndef PARAMETERIZABLE_H
#define PARAMETERIZABLE_H

#include "eigen/eigen_realtime_check.hpp" // Include this before Eigen header files


#include <set>
#include <string>
#include <vector>
#include <eigen3/Eigen/Core>



namespace DmpBbo {

/** \brief Class for providing access to a model's parameters as a vector.
 *
 * Different function approximators have different types of model parameters. For instance, LWR
 * has the centers and widths of basis functions, along with the slopes of each line segment.
 * Parameterizable::getValues provides a means to access these parameter as one vector.
 *
 * This may be useful for instance when optimizing the model parameters with black-box
 * optimization, which is agnostic about the semantics of the model parameters. 
 */
class Parameterizable {
  
public: 
  
  /** Destructor */
  virtual ~Parameterizable(void) {};
  
  /** Get the size of the vector of selected parameters, as returned by getParameterVectorSelected(
   * \return The size of the vector representation of the selected parameters.
   */
  virtual int getParameterVectorSelectedSize(void) const;
  
  /**
   * Get the values of the selected parameters in one vector.
   * \param[out] values The selected parameters concatenated in one vector
   * \param[in] normalized Whether to normalize the data or not
   */
  virtual void getParameterVectorSelected(Eigen::VectorXd& values, bool normalized=false) const;
  
  /**
   * Get the normalized values of the selected parameters in one vector.
   * \param[out] values The selected parameters concatenated in one vector
   */
  virtual void getParameterVectorSelectedNormalized(Eigen::VectorXd& values) const {
    getParameterVectorSelected(values, true);
  }

  /**
   * Get the minimum and maximum of the selected parameters in one vector.
   * \param[out] min The minimum of the selected parameters concatenated in one vector
   * \param[out] max The minimum of the selected parameters concatenated in one vector
   */
  void getParameterVectorSelectedMinMax(Eigen::VectorXd& min, Eigen::VectorXd& max) const;

  /**
   * Get the minimum and maximum values of the current parameter vector.
   * \param[out] min Minimum values of the parameters vector
   * \param[out] max Maximum values of the parameters vector
   */
  void getParameterVectorAllMinMax(Eigen::VectorXd& min, Eigen::VectorXd& max) const;
  
  /**
   * Get the ranges of the selected parameters, i.e. max-min, in one vector.
   * \param[out] ranges The ranges of the selected parameters concatenated in one vector
   */
   inline void getParameterVectorSelectedRanges(Eigen::VectorXd& ranges) const {
     Eigen::VectorXd min, max;
     getParameterVectorSelectedMinMax(min, max);
     ranges = (max.array()-min.array());
   }
  
  /**
   * Set all the values of the selected parameters with one vector.
   * \param[in] values The new values of the selected parameters in one vector
   * \param[in] normalized Whether the data is normalized or not
   */
  virtual void setParameterVectorSelected(const Eigen::VectorXd& values, bool normalized=false);
  
  /**
   * Set all the values of the selected parameters with one vector of normalized values.
   * \param[in] values The new values of the selected parameters in one vector of normalized values.
   */
  virtual void setParameterVectorSelectedNormalized(const Eigen::VectorXd& values) {
    setParameterVectorSelected(values, true);
  }

  /**
   * Determine which subset of parameters is represented in the vector returned by Parameterizable::getParameterVectorSelected
   * 
   * Different function approximators have different types of model parameters. For instance, LWR
   * has the centers and widths of basis functions, along with the slopes of each line segment.
   * Parameterizable::setSelectedParameters provides a means to determine which parameters 
   * should be returned by Parameterizable::getParameterVectorSelected, i.e. by calling:
   *   std::set<std::string> selected;
   *   selected.insert("slopes");
   *   model_parameters.setSelectedParameters(selected)
   * \param[in] selected_values_labels The names of the parameters that are selected
   */
  virtual void setSelectedParameters(const std::set<std::string>& selected_values_labels);

  /** Set the parameters that are currently selected. 
   * Convenience function that allow only a string to be passed, rather than a set of strings.
   * \param[in] selected The name of the parameters that are selected
   */
  void setSelectedParametersOne(std::string selected)
  {
    std::set<std::string> selected_set;
    selected_set.insert(selected);
    setSelectedParameters(selected_set);    
  }
  
  /** Return all the names of the parameter types that can be selected.
   * \param[out] selected_values_labels Names of the parameter types that can be selected
   */
  virtual void getSelectableParameters(std::set<std::string>& selected_values_labels) const = 0;

  /**
   * Get a mask for selecting parameters.
   * 
   * \param[in] selected_values_labels Labels of the selected parameter values
   * \param[out] selected_mask A mask indicating indices of selected parameters. 0 indicates not selected, >0 indicates selected.
   *
   * For instance, if the parameters consists of centers, widths and slopes the 
   * parameter values vector will be something like
\verbatim
    centers     widths    slopes
[ 100 110 120 10 10 10 0.4 0.7 0.4 ]
\endverbatim
   * In this case, if selected_values_labels contains "centers" and "slopes", the mask will be:
\verbatim
    centers     widths    slopes
[   1   1   1  0  0  0   3   3   3 ]
\endverbatim
   * The '0' indicates that these parameters are not selected.
   * The other ones have different numbers so that they may be discerned from one another (as
   * required in Parameterizable::getParameterVectorSelectedMinMax for instance.
   */
  virtual void getParameterVectorMask(const std::set<std::string> selected_values_labels, Eigen::VectorXi& selected_mask) const = 0;
  
  /** Get the size of the parameter values vector when it contains all available
   * parameter values.
   * \return The size of the parameter vector
   *
   * For instance, if the parameters consists of centers, widths and slopes the 
   * parameter values vector will be something like
\verbatim
    centers     widths    slopes
[ 100 110 120 10 10 10 0.4 0.7 0.4 ]
\endverbatim
   * then getParameterVectorAllSize() will return 9 
   */
  virtual int getParameterVectorAllSize(void) const = 0;
  
  /** Return a vector that returns all available parameter values.
   * \param[out] values All available parameter values in one vector.
   * \remarks Contrast this with Parameterizable::getParameterVectorSelected, which return only
   * the SELECTED parameter values. Selecting parameters is done with
   * Parameterizable::setSelectedParameters
   */
  virtual void getParameterVectorAll(Eigen::VectorXd& values) const = 0;
  
  //void getParameterVectorAllNormalized(Eigen::VectorXd& values_normalized) const;

  /** Set all available parameter values with one vector.
   * \param[in] values All available parameter values in one vector.
   * \remarks Contrast this with Parameterizable::setParameterVectorSelected, which sets only
   * the SELECTED parameter values. Selecting parameters is done with
   * Parameterizable::setSelectedParameters
   */
  virtual void setParameterVectorAll(const Eigen::VectorXd& values) = 0;
    
  /** Turn certain modifiers on or off.
   *
   * This can be used to modify exactly what is returned by Parameterizable::getParameterVectorAll(). 
   * For an example, see ModelParametersLWR::setParameterVectorModifierPrivate()
   *
   * This function calls the virtual private function Parameterizable::setParameterVectorModifierPrivate(), which may (but must not be) overridden by subclasses of Parameterizable.
   *
   * \param[in] modifier The name of the modifier
   * \param[in] new_value Whether to turn the modifier on (true) or off (false)
   */
  void setParameterVectorModifier(std::string modifier, bool new_value);
  
  /** The vector (VectorXd) with parameter values can be split into different parts (as vector<VectorXd>; this function specifices the length of each sub-vector.
   * 
   * For instance if the parameter vector is of length 12, getParameterVectorSelected(VectorXd) would return a VectorXd of size 12.
   * If you would like these 16 values to be split into 4 VectorXd of length 3, you would set 
   * setVectorLengthsPerDimension([3 3 3 3]).
   * getParameterVectorSelected(VectorXd) would still return a VectorXd of size 12, but getParameterVectorSelected(std::vector<Eigen::VectorXd>&) would return a std::vector of length 4, with each VectorXd of size 3.
   *
   * This is a convenience function to be able to use vector<VectorXd> instead of VectorXd when getting/setting parameter values.
   *
   * \param[in] lengths_per_dimension The length of each vector in each dimension.
   */
  void setVectorLengthsPerDimension(const Eigen::VectorXi& lengths_per_dimension)
  {
    assert(lengths_per_dimension.sum()==getParameterVectorSelectedSize());
    lengths_per_dimension_ = lengths_per_dimension;
  }

  /** Get the specified length of each vector in each dimension.
   * \see setVectorLengthsPerDimension()
   * \return The length of each vector in each dimension.
   */
  Eigen::VectorXi getVectorLengthsPerDimension(void) const
  {
    return lengths_per_dimension_;
  }
  
  /**
   * Get the values of the selected parameters in one vector.
   * \param[out] values The selected parameters concatenated in one vector
   * \param[in] normalized Whether to normalize the data or not
   * \remarks The lenghts of each Eigen::VectorXd in the std::vector is set with Parameterizable::setVectorLengthsPerDimension()
   */
  void getParameterVectorSelected(std::vector<Eigen::VectorXd>& values, bool normalized=false) const;
  
  /**
   * Set all the values of the selected parameters with a vector of vectors.
   * \param[in] values The new values of the selected parameters in one vector
   * \param[in] normalized Whether the data is normalized or not
   * \remarks The lenghts of each Eigen::VectorXd in the std::vector is set with Parameterizable::setVectorLengthsPerDimension()
   */
  void setParameterVectorSelected(const std::vector<Eigen::VectorXd>& values, bool normalized=false);

  
private:
  /** Turn certain modifiers on or off, see Parameterizable::setParameterVectorModifier().
   *
   * Parameterizable::setParameterVectorModifierPrivate(), This function may (but must not be) overridden by subclasses of Parameterizable, depending on whether the subclass has modifiers (or not)
   *
   * \param[in] modifier The name of the modifier
   * \param[in] new_value Whether to turn the modifier on (true) or off (false)
   */
  virtual void setParameterVectorModifierPrivate(std::string modifier, bool new_value)
  {
    // Can be overridden by subclasses
  }
  
  Eigen::VectorXi selected_mask_;

  /** 
   * \see Parameterizable::setVectorLengthsPerDimension()
   */
  Eigen::VectorXi lengths_per_dimension_;
  
  // Since this is a cached variable, it needs to be mutable so that const functions may change it.
  mutable Eigen::VectorXd parameter_vector_all_initial_;

};

}

#endif
