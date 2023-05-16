#!/bin/bash

D=results_test

################################################################################
# STEP 1: Train the DMP with a trajectory. Try it with different # basis functions
python3 step1_train_dmp_from_trajectory_file.py trajectories/trajectory_cart_rarm.txt trajectories/trajectory_cart_larm.txt ${D}/training --n 15
python3 step1_train_dmp_from_trajectory_file.py trajectories/trajectory_cart_rarm_optimal.txt trajectories/trajectory_cart_larm_optimal.txt ${D}/training --n 20
# 10 basis functions look good; choose it as initial DMP for optimization
cp ${D}/training/dmp_rarm_trained_20.json ${D}/dmp_rarm_initial.json
cp ${D}/training/dmp_larm_trained_20.json ${D}/dmp_larm_initial.json

################################################################################
# STEP 2: Define and save the task
python3 step2_define_task.py ${D} task.json trajectories/trajectory.txt


################################################################################
# STEP 3: Tune the exploration noise

# Exploration noise
python3 step3_tune_exploration.py ${D}/dmp_rarm_initial.json ${D}/dmp_larm_initial.json ${D}/tune_exploration --save --n 30 --sigma 1.0
# DU="${D}/tune_exploration/sigma_1.000"
# for i_sample in $(seq -f "%02g" 0 9)
# do # Run the sampled DMPs on the robot
#   # ../../bin/robotExecuteDmp ${DU}/${i_sample}_dmp_for_cpp.json ${DU}/${i_sample}_cost_vars.txt
#   python3 robotExecuteDmpReaching.py ${DU}/${i_sample}_dmp.json ${DU}/${i_sample}_cost_vars.txt
# done
# python3 plot_rollouts.py ${DU} ${D}/task.json --save # Save the results as a png

# # Medium exploration noise
# python3 step3_tune_exploration.py ${D}/dmp_initial.json ${D}/tune_exploration --save --n 10 --sigma  20.0
# DU="${D}/tune_exploration/sigma_20.000"
# for i_sample in $(seq -f "%02g" 0 9)
# do # Run the sampled DMPs on the robot
#   ../../bin/robotExecuteDmp ${DU}/${i_sample}_dmp_for_cpp.json ${DU}/${i_sample}_cost_vars.txt
# done
# python3 plot_rollouts.py ${DU} ${D}/task.json --save # Save the results as a png

# # High exploration noise
# python3 step3_tune_exploration.py ${D}/dmp_initial.json ${D}/tune_exploration --save --n 10 --sigma 40.0
# DU="${D}/tune_exploration/sigma_40.000"
# for i_sample in $(seq -f "%02g" 0 9)
# do # Run the sampled DMPs on the robot
#   ../../bin/robotExecuteDmp ${DU}/${i_sample}_dmp_for_cpp.json ${DU}/${i_sample}_cost_vars.txt
# done
# python3 plot_rollouts.py ${DU} ${D}/task.json --save # Save the results as a png


# Initial distribution
cp ${D}/tune_exploration/sigma_1.000/distribution_rarm.json ${D}/distribution_initial_rarm.json
cp ${D}/tune_exploration/sigma_1.000/distribution_larm.json ${D}/distribution_initial_larm.json


################################################################################
# STEP 4: Prepare the optimization
python3 step4_prepare_optimization.py ${D} ${D}/updates_rarm ${D}/updates_larm


################################################################################
# STEP 5: Run the optimization
for i_update in $(seq -f "%05g" 0 149)
do
  
  # Run the sampled DMPs on the robot
  DU_rarm="${D}/updates_rarm/update${i_update}"
  DU_larm="${D}/updates_larm/update${i_update}"

  # Evaluation rollout
  # ../../bin/robotExecuteDmp ${DU}/eval_dmp_for_cpp.json ${DU}/eval_cost_vars.txt
  python3 robotExecuteDmpReaching.py ${DU_rarm}/eval_dmp.json ${DU_larm}/eval_dmp.json ${DU_rarm}/eval_cost_vars.txt ${DU_larm}/eval_cost_vars.txt
  # Samples rollouts
  for i in $(seq -f "%03g" 0 29)
  do
    # ../../bin/robotExecuteDmp ${DU}/${i}_dmp_for_cpp.json ${DU}/${i}_cost_vars.txt
    python3 robotExecuteDmpReaching.py ${DU_rarm}/${i}_dmp.json ${DU_larm}/${i}_dmp.json ${DU_rarm}/${i}_cost_vars.txt ${DU_larm}/${i}_cost_vars.txt
  done
  
  # Update the distribution (given the cost_vars above), and generate the
  # next batch of samples
  python3 step5_one_optimization_update.py ${D}/updates_rarm ${i_update}
  python3 step5_one_optimization_update.py ${D}/updates_larm ${i_update}

  
done
  
python3 plot_optimization.py ${D} --save