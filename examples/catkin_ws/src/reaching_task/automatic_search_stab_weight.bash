R=automatic_search_stab_weight_0_120

while :
do
    midpoint=$(head -n 1 ${R}/midpoint.txt)
    min_weight=$(head -n 1 ${R}/min_weight.txt)
    max_weight=$(head -n 1 ${R}/max_weight.txt)

    echo "Midpoint = ${midpoint}"
    D=${R}/results_w_stab_${midpoint}/
    D_min=${R}/results_w_stab_${min_weight}/
    D_max=${R}/results_w_stab_${max_weight}/

    # Step 1: train dmp
    python3 step1_train_dmp_from_trajectory_file.py trajectories/trajectory.txt ${D}/training --n 20 --save
    cp ${D}/training/dmp_trained_20.json ${D}/dmp_initial.json

    # Step 2: define task with cost weights
    python3 step2_define_task.py ${D} task.json trajectories/trajectory.txt ${R}/midpoint.txt

    # Step 3: define exploration matrix
    python3 step3_tune_exploration.py ${D}/dmp_initial.json ${D}/tune_exploration --save --n 30 --sigma   1.0
    cp ${D}/tune_exploration/sigma_1.000/distribution.json ${D}/distribution_initial.json

    # Step 4: prepare the optimization
    python3 step4_prepare_optimization.py ${D}

    # Step 5: run the learning
    for i_update in $(seq -f "%05g" 0 99) # TEST THE SCRIPT WITH 3 UPDATES FIRST!!!!!!!!!!!!!!
    do
    
    # Run the sampled DMPs on the robot
    DU="${D}/update${i_update}"
    # Evaluation rollout
    # ../../bin/robotExecuteDmp ${DU}/eval_dmp_for_cpp.json ${DU}/eval_cost_vars.txt
    python3 robotExecuteDmpReaching.py ${DU}/eval_dmp.json ${DU}/eval_cost_vars.txt
    # Samples rollouts
    for i in $(seq -f "%03g" 0 29)
    do
        # ../../bin/robotExecuteDmp ${DU}/${i}_dmp_for_cpp.json ${DU}/${i}_cost_vars.txt
        python3 robotExecuteDmpReaching.py ${DU}/${i}_dmp.json ${DU}/${i}_cost_vars.txt
    done
    
    # Update the distribution (given the cost_vars above), and generate the
    # next batch of samples
    python3 step5_one_optimization_update.py ${D} ${i_update}
    
    done

    # Step 6: calculate new midpoint
    python3 calculate_midpoint.py ${D_min} ${D_max} ${D}
done
