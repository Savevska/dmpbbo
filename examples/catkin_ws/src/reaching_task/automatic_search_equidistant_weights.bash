R=automatic_search_stab_weight_equidistant_0_60
# weights=(5.0, 10.0 15.0 20.0 25.0 35.0 40.0 50.0 55.0)
for weight in 10.0 15.0 20.0 25.0 35.0 40.0 50.0 55.0
do
    echo "Stability weight = ${weight}"
    D=${R}/results_w_stab_${weight}/

    # Step 1: train dmp
    python3 step1_train_dmp_from_trajectory_file.py trajectories/trajectory.txt ${D}/training --n 20 --save
    cp ${D}/training/dmp_trained_20.json ${D}/dmp_initial.json

    # Step 2: define task with cost weights
    python3 step2_define_task.py ${D} task.json trajectories/trajectory.txt ${weight}

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

done