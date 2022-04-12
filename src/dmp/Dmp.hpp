/**
 * @file Dmp.hpp
 * @brief  Dmp class header file.
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

#ifndef _DMP_H_
#define _DMP_H_

#include <nlohmann/json_fwd.hpp>
#include <set>

#include "dynamicalsystems/DynamicalSystem.hpp"

namespace DmpBbo {

// forward declaration
class FunctionApproximator;
class SpringDamperSystem;
class Trajectory;

/** \defgroup Dmps Dynamic Movement Primitives Module
 */

/**
 * \brief Implementation of Dynamical Movement Primitives.
 * \ingroup Dmps
 */
class Dmp : public DynamicalSystem {
 public:
  /**
   *  Initialization constructor.
   *  \param tau             Time constant
   *  \param y_init          Initial state
   *  \param y_attr          Attractor state
   *  \param alpha_spring_damper \f$\alpha\f$ in the spring-damper system of the
   * dmp \param goal_system     Dynamical system to compute delayed goal \param
   * phase_system    Dynamical system to compute the phase \param gating_system
   * Dynamical system to compute the gating term \param function_approximators
   * Function approximators for the forcing term \param scaling         Which
   * method to use for scaling the forcing term ("NO_SCALING",
   * "G_MINUS_Y0_SCALING", "AMPLITUDE_SCALING")
   */
  Dmp(double tau, Eigen::VectorXd y_init, Eigen::VectorXd y_attr,
      std::vector<FunctionApproximator*> function_approximators,
      double alpha_spring_damper, DynamicalSystem* goal_system,
      DynamicalSystem* phase_system, DynamicalSystem* gating_system,
      std::string scaling = "NO_SCALING");

  /**
   *  Initialization constructor for Dmps of known dimensionality, but with
   * unknown initial and attractor states. \param n_dims_dmp      Dimensionality
   * of the DMP \param alpha_spring_damper \f$\alpha\f$ in the spring-damper
   * system of the dmp \param goal_system     Dynamical system to compute
   * delayed goal \param phase_system    Dynamical system to compute the phase
   *  \param gating_system   Dynamical system to compute the gating term
   *  \param function_approximators Function approximators for the forcing term
   *  \param scaling         Which method to use for scaling the forcing term
   * ("NO_SCALING", "G_MINUS_Y0_SCALING", "AMPLITUDE_SCALING")
   */
  Dmp(int n_dims_dmp, std::vector<FunctionApproximator*> function_approximators,
      double alpha_spring_damper, DynamicalSystem* goal_system,
      DynamicalSystem* phase_system, DynamicalSystem* gating_system,
      std::string scaling = "NO_SCALING");

  /**
   *  Constructor that initializes the DMP with default dynamical systems.
   *  \param tau       Time constant
   *  \param y_init    Initial state
   *  \param y_attr    Attractor state
   *  \param function_approximators Function approximators for the forcing term
   *  \param dmp_type  The type of DMP ("IJSPEERT_2002_MOVEMENT",
   * "KULVICIUS_2012_JOINING", "COUNTDOWN_2013") \param scaling         Which
   * method to use for scaling the forcing term ("NO_SCALING",
   * "G_MINUS_Y0_SCALING", "AMPLITUDE_SCALING")
   */
  Dmp(double tau, Eigen::VectorXd y_init, Eigen::VectorXd y_attr,
      std::vector<FunctionApproximator*> function_approximators,
      std::string dmp_type = "KULVICIUS_2012_JOINING",
      std::string scaling = "NO_SCALING");

  /**
   *  Initialization constructor for Dmps of known dimensionality, but with
   * unknown initial and attractor states. Initializes the DMP with default
   * dynamical systems. \param n_dims_dmp      Dimensionality of the DMP \param
   * function_approximators Function approximators for the forcing term \param
   * dmp_type  The type of DMP ("IJSPEERT_2002_MOVEMENT",
   * "KULVICIUS_2012_JOINING", "COUNTDOWN_2013") \param scaling         Which
   * method to use for scaling the forcing term ("NO_SCALING",
   * "G_MINUS_Y0_SCALING", "AMPLITUDE_SCALING")
   */
  Dmp(int n_dims_dmp, std::vector<FunctionApproximator*> function_approximators,
      std::string dmp_type = "KULVICIUS_2012_JOINING",
      std::string scaling = "NO_SCALING");

  /**
   *  Initialization constructor for Dmps without a forcing term.
   *  \param tau             Time constant
   *  \param y_init          Initial state
   *  \param y_attr          Attractor state
   *  \param alpha_spring_damper \f$\alpha\f$ in the spring-damper system of the
   * dmp \param goal_system     Dynamical system to compute delayed goal
   */
  Dmp(double tau, Eigen::VectorXd y_init, Eigen::VectorXd y_attr,
      double alpha_spring_damper, DynamicalSystem* goal_system);

  /** Destructor. */
  ~Dmp(void);

  virtual void integrateStart(Eigen::Ref<Eigen::VectorXd> x,
                              Eigen::Ref<Eigen::VectorXd> xd) const;

  void differentialEquation(const Eigen::Ref<const Eigen::VectorXd>& x,
                            Eigen::Ref<Eigen::VectorXd> xd) const;

  /**
   * Return analytical solution of the system at certain times (and return
   * forcing terms)
   *
   * \param[in]  ts  A vector of times for which to compute the analytical
   * solutions \param[out] xs  Sequence of state vectors. T x D or D x T matrix,
   * where T is the number of times (the length of 'ts'), and D the size of the
   * state (i.e. dim()) \param[out] xds Sequence of state vectors (rates of
   * change). T x D or D x T matrix, where T is the number of times (the length
   * of 'ts'), and D the size of the state (i.e. dim()) \param[out]
   * forcing_terms The forcing terms for each dimension, for debugging purposes
   * only. \param[out] fa_output The output of the function approximators, for
   * debugging purposes only.
   *
   * \remarks The output xs and xds will be of size D x T \em only if the matrix
   * x you pass as an argument of size D x T. In all other cases (i.e. including
   * passing an empty matrix) the size of x will be T x D. This feature has been
   * added so that you may pass matrices of either size.
   */
  void analyticalSolution(const Eigen::VectorXd& ts, Eigen::MatrixXd& xs,
                          Eigen::MatrixXd& xds, Eigen::MatrixXd& forcing_terms,
                          Eigen::MatrixXd& fa_output) const;

  /**
   * Return analytical solution of the system at certain times (and return
   * forcing terms)
   *
   * \param[in]  ts  A vector of times for which to compute the analytical
   * solutions \param[out] xs  Sequence of state vectors. T x D or D x T matrix,
   * where T is the number of times (the length of 'ts'), and D the size of the
   * state (i.e. dim()) \param[out] xds Sequence of state vectors (rates of
   * change). T x D or D x T matrix, where T is the number of times (the length
   * of 'ts'), and D the size of the state (i.e. dim()) \param[out]
   * forcing_terms The forcing terms for each dimension, for debugging purposes
   * only.
   *
   * \remarks The output xs and xds will be of size D x T \em only if the matrix
   * x you pass as an argument of size D x T. In all other cases (i.e. including
   * passing an empty matrix) the size of x will be T x D. This feature has been
   * added so that you may pass matrices of either size.
   */
  inline void analyticalSolution(const Eigen::VectorXd& ts, Eigen::MatrixXd& xs,
                                 Eigen::MatrixXd& xds,
                                 Eigen::MatrixXd& forcing_terms) const
  {
    Eigen::MatrixXd fa_output;
    analyticalSolution(ts, xs, xds, forcing_terms, fa_output);
  }

  inline void analyticalSolution(const Eigen::VectorXd& ts, Eigen::MatrixXd& xs,
                                 Eigen::MatrixXd& xds) const
  {
    Eigen::MatrixXd forcing_terms, fa_output;
    analyticalSolution(ts, xs, xds, forcing_terms, fa_output);
  }

  /**
   * Return analytical solution of the system at certain times
   *
   * \param[in]  ts  A vector of times for which to compute the analytical
   * solutions \param[out] trajectory The computed states as a trajectory.
   */
  virtual void analyticalSolution(const Eigen::VectorXd& ts,
                                  Trajectory& trajectory) const
  {
    Eigen::MatrixXd xs, xds;
    analyticalSolution(ts, xs, xds);
    statesAsTrajectory(ts, xs, xds, trajectory);
  }

  /**
   * Return analytical solution of the system at certain times
   *
   * \param[in]  ts  A vector of times for which to compute the analytical
   * solutions \param[out] trajectory The computed states as a trajectory.
   * \param[out] forcing_terms The forcing terms
   */
  inline void analyticalSolution(const Eigen::VectorXd& ts,
                                 Trajectory& trajectory,
                                 Eigen::MatrixXd& forcing_terms) const
  {
    Eigen::MatrixXd xs, xds;
    analyticalSolution(ts, xs, xds, forcing_terms);
    statesAsTrajectory(ts, xs, xds, trajectory);
  }

  /** Get the output of a DMP dynamical system as a trajectory.
   *  As a dynamical system, the state vector of a DMP contains the output of
   * the goal, spring, phase and gating system. What we are most interested in
   * is the output of the spring system. This function extracts that
   * information, and also computes the accelerations of the spring system,
   * which are only stored implicitely in xd_in because second order systems are
   * converted to first order systems with expanded state.
   *
   * \param[in] x_in  State vector over time (size n_time_steps X dim())
   * \param[in] xd_in State vector over time (rates of change)
   * \param[out] y_out  State vector over time (size n_time_steps X dim_orig())
   * \param[out] yd_out  State vector over time (rates of change)
   * \param[out] ydd_out  State vector over time (rates of change of rates of
   * change)
   *
   */
  virtual void statesAsTrajectory(const Eigen::MatrixXd& x_in,
                                  const Eigen::MatrixXd& xd_in,
                                  Eigen::MatrixXd& y_out,
                                  Eigen::MatrixXd& yd_out,
                                  Eigen::MatrixXd& ydd_out) const;

  /** Get the output of a DMP dynamical system as a trajectory.
   *  As a dynamical system, the state vector of a DMP contains the output of
   * the goal, spring, phase and gating system. What we are most interested in
   * is the output of the spring system. This function extracts that
   * information, and also computes the accelerations of the spring system,
   * which are only stored implicitely in xd_in because second order systems are
   * converted to first order systems with expanded state.
   *
   * \param[in] ts    A vector of times
   * \param[in] x_in  State vector over time
   * \param[in] xd_in State vector over time (rates of change)
   * \param[out] trajectory Trajectory representation of the DMP state vector
   * output.
   *
   */
  virtual void statesAsTrajectory(const Eigen::VectorXd& ts,
                                  const Eigen::MatrixXd& x_in,
                                  const Eigen::MatrixXd& xd_in,
                                  Trajectory& trajectory) const;

  /**
   * Accessor function for the time constant.
   * \param[in] tau Time constant
   * We need to override DynamicalSystem::set_tau, because the DMP must also
   * change the time constant of all of its subsystems.
   */
  virtual void set_tau(double tau);

  /** Accessor function for the initial state of the system.
   *  \param[in] y_init Initial state of the system.
   * We need to override DynamicalSystem::set_initial_state, because the DMP
   * must also change the initial state  of the goal system as well.
   */
  virtual void set_initial_state(const Eigen::VectorXd& y_init);

  /** Accessor function for the attractor state of the system.
   *  \param[in] y_attr Attractor state of the system.
   */
  virtual void set_attractor_state(const Eigen::VectorXd& y_attr);

  /**
   * Accessor function for damping coefficient of spring-damper system
   * \param[in] damping_coefficient Damping coefficient
   */
  void set_damping_coefficient(double damping_coefficient);

  /**
   * Accessor function for spring constant of spring-damper system
   * \param[in] spring_constant Spring constant
   */
  void set_spring_constant(double spring_constant);

  /** Given a trajectory, compute the inputs and targets for the function
   * approximators. For a standard Dmp (such as the one in this class) the
   * inputs will be the phase over time, and the targets will be the forcing
   * term (with the gating function factored out). \param[in] trajectory
   * Trajectory, e.g. a demonstration. \param[out] fa_inputs_phase The inputs
   * for the function approximators (phase signal) \param[out] fa_targets The
   * targets for the function approximators (forcing term)
   */
  void computeFunctionApproximatorInputsAndTargets(
      const Trajectory& trajectory, Eigen::VectorXd& fa_inputs_phase,
      Eigen::MatrixXd& fa_targets) const;

  /** Compute the outputs of the function approximators.
   * \param[in] phase_state The phase states for which the outputs are computed.
   * \param[out] fa_output The outputs of the function approximators.
   */
  virtual void computeFunctionApproximatorOutput(
      const Eigen::Ref<const Eigen::MatrixXd>& phase_state,
      Eigen::MatrixXd& fa_output) const;

  /** Get a pointer to the function approximator for a certain dimension.
   * \param[in] i_dim Dimension for which to get the function approximator
   * \return Pointer to the function approximator.
   */
  inline FunctionApproximator* function_approximator(int i_dim) const
  {
    assert(i_dim < (int)function_approximators_.size());
    return function_approximators_[i_dim];
  }

  /** Read an object from json.
   *  \param[in]  j json input
   *  \param[out] obj The object read from json
   *
   * See also: https://github.com/nlohmann/json/issues/1324
   */
  friend void from_json(const nlohmann::json& j, Dmp*& obj);

  /** Write an object to json.
   *  \param[in] obj The object to write to json
   *  \param[out]  j json output
   *
   * See also:
   *   https://github.com/nlohmann/json/issues/1324
   *   https://github.com/nlohmann/json/issues/716
   */
  inline friend void to_json(nlohmann::json& j, const Dmp* const& obj)
  {
    obj->to_json_helper(j);
  }

 private:
  /** Write this object to json.
   *  \param[out]  j json output
   *
   * See also:
   *   https://github.com/nlohmann/json/issues/1324
   *   https://github.com/nlohmann/json/issues/716
   */
  void to_json_helper(nlohmann::json& j) const;

  /** @name Linear closed loop controller
   *  @{
   */
  /** Delayed goal system. Also see \ref sec_delayed_goal */
  DynamicalSystem* goal_system_;
  /** Spring-damper system. Also see \ref page_dmp */
  SpringDamperSystem* spring_system_;
  /** @} */  // end of group_linear

  /** @name Non-linear open loop controller
   *  @{
   */
  /** System that determined the phase of the movement. */
  DynamicalSystem* phase_system_;
  /** System to gate the output of the function approximators. Starts at 1 and
   * converges to 0. */
  DynamicalSystem* gating_system_;

  /** The function approximators, one for each dimension, in the forcing term.
   */
  std::vector<FunctionApproximator*> function_approximators_;

  /** How is the forcing term scaled? */
  std::string forcing_term_scaling_;

  /** Ranges of the trajectory (per dimension) for (optional) scaling of forcing
   * term.  */
  Eigen::VectorXd trajectory_amplitudes_;

  /** @} */  // end of group_nonlinear

  /** Pre-allocated memory to avoid allocating it during run-time. To enable
   * real-time. */
  mutable Eigen::VectorXd attractor_state_prealloc_;

  /** Pre-allocated memory to avoid allocating it during run-time. To enable
   * real-time. */
  mutable Eigen::VectorXd initial_state_prealloc_;

  /** Pre-allocated memory to avoid allocating it during run-time. To enable
   * real-time. */
  mutable Eigen::MatrixXd fa_outputs_one_prealloc_;

  /** Pre-allocated memory to avoid allocating it during run-time. To enable
   * real-time. */
  mutable Eigen::MatrixXd fa_outputs_prealloc_;

  /** Pre-allocated memory to avoid allocating it during run-time. To enable
   * real-time. */
  mutable Eigen::MatrixXd fa_output_prealloc_;

  /** Pre-allocated memory to avoid allocating it during run-time. To enable
   * real-time. */
  mutable Eigen::VectorXd forcing_term_prealloc_;

  /** Pre-allocated memory to avoid allocating it during run-time. To enable
   * real-time. */
  mutable Eigen::VectorXd g_minus_y0_prealloc_;

  /**
   *  Helper function for constructor.
   *  \param spring_system   Spring-damper system                 cf.
   * Dmp::spring_system_ \param goal_system     System to compute delayed goal,
   * cf. Dmp::damping_coefficient_ \param phase_system    System to compute the
   * phase,         cf. Dmp::phase_system_ \param gating_system   System to
   * compute the gating term,   cf. Dmp::gating_system_ \param
   * function_approximators Function approximators for the forcing term, cf.
   * Dmp::function_approximators_
   */
  void initSubSystems(double alpha_spring_system, DynamicalSystem* goal_system,
                      DynamicalSystem* phase_system,
                      DynamicalSystem* gating_system);

  void initSubSystems(std::string dmp_type);

  void initFunctionApproximators(
      std::vector<FunctionApproximator*> function_approximators);
};

}  // namespace DmpBbo

#endif  // _DMP_H_

namespace DmpBbo {

/** \page page_dmp Dynamical Movement Primitives

This page provides an  overview of the implementation of DMPs in the \c dmps/
module.

It is assumed you have read about the theory behind DMPs in the tutorial <a
href="https://github.com/stulp/dmpbbo/blob/master/tutorial/dmp.md">tutorial/dmp.md</a>.
Note that in the tutorial, we have used the notation \f$[z~y]\f$ for consistency
with the DMP literature. In the C++ implementation, the order is rather
\f$[y~z]\f$.


Since a Dynamical Movement Primitive is a dynamical system, the Dmp class
derives from the DynamicalSystem class. It overrides the virtual function
DynamicalSystem::integrateStart(). Integrating the DMP numerically (Euler or 4th
order Runge-Kutta) is done with the generic DynamicalSystem::integrateStep()
function. It also implements the pure virtual function
DynamicalSystem::analyticalSolution(). Because a DMP cannot be solved
analytically (we cannot write it in closed form due to the arbitrary forcing
term), calling Dmp::analyticalSolution() in fact performs a numerical Euler
integration (although the linear subsystems (phase, gating, etc.) are
analytically solved because this is faster computationally).


\em Remark. Dmp inherits the function DynamicalSystem::integrateStep() from the
DynamicalSystem class. DynamicalSystem::integrateStep() uses either Euler
integration, or 4-th order Runge-Kutta.  The latter is more accurate, but
requires 4 calls of DynamicalSystem::differentialEquation() instead of 1). Which
one is used can be set with DynamicalSystem::set_integration_method(). To
numerically integrate a dynamical system, one must carefully choose the
integration time dt. Choosing it too low leads to inaccurate integration, and
the numerical integration will diverge from the 'true' solution acquired through
analytical solution. See http://en.wikipedia.org/wiki/Euler%27s_method for
examples. Choosing dt depends entirely on the time-scale (seconds vs. years) and
parameters of the dynamical system (time constant, decay parameters). For DMPs,
which are expected to take between 0.5-10 seconds, dt is usually chosen to be in
the range 0.01-0.001.


*/

}
