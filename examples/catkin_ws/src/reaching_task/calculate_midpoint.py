import argparse
import os
import sys
sys.path.append("/home/user/talos_ws/dmpbbo")
import dmpbbo.json_for_cpp as jc
import numpy as np
import pandas as pd

def calculate_costs(directory):
    updates = np.sort(os.listdir(directory))
    costs = []
    for update in updates:
        if "update0" in update and update!="update00100":
            files = os.listdir(os.path.join(os.getcwd(), directory + update))
            for f in files:
                if "eval_costs.txt" in f:
                    c = np.loadtxt(os.path.join(os.getcwd(), directory + update) + "/" + f)
                    costs.append(c)
    res = pd.DataFrame(columns=["cost", "stab_cost", "goal_cost", "orient_cost"], data=costs)
    return res

def main():
    """ Main function that is called when executing the script. """

    parser = argparse.ArgumentParser()
    parser.add_argument("min_limit_dir", help="directory with the min stability weight")
    parser.add_argument("max_limit_dir", help="directory with the min stability weight")
    parser.add_argument("midpoint_dir", help="directory with the middle stability weight")

    args = parser.parse_args()

    min_task = jc.loadjson(args.min_limit_dir + "/task.json")
    max_task = jc.loadjson(args.max_limit_dir + "/task.json")
    mid_task = jc.loadjson(args.midpoint_dir + "/task.json")

    min_weight = min_task.stability_weight_
    max_weight = max_task.stability_weight_
    mid_weight = mid_task.stability_weight_

    if min_weight == 0:
        min_weight_ = 1.0
    else:
        min_weight_ = min_weight

    costs_min_limit = calculate_costs(args.min_limit_dir)
    costs_max_limit = calculate_costs(args.max_limit_dir)
    costs_midpoint = calculate_costs(args.midpoint_dir)

    avg_cost_min = (costs_min_limit["stab_cost"].iloc[-50:]/min_weight_).mean() + \
                   (costs_min_limit["goal_cost"].iloc[-50:]/min_task.goal_weight_).mean() + \
                   (costs_min_limit["orient_cost"].iloc[-50:]/min_task.goal_orientation_weight_).mean()
    
    avg_cost_max = (costs_max_limit["stab_cost"].iloc[-50:]/max_weight).mean() + \
                    (costs_max_limit["goal_cost"].iloc[-50:]/max_task.goal_weight_).mean() + \
                    (costs_max_limit["orient_cost"].iloc[-50:]/max_task.goal_orientation_weight_).mean()
    
    avg_cost_mid = (costs_midpoint["stab_cost"].iloc[-50:]/mid_weight).mean() + \
                    (costs_midpoint["goal_cost"].iloc[-50:]/mid_task.goal_weight_).mean() + \
                    (costs_midpoint["orient_cost"].iloc[-50:]/mid_task.goal_orientation_weight_).mean()

    max_position_error = 5e-3
    max_orientation_error = 1e-2

    # calculate new midpoint
    upper_half = avg_cost_mid - avg_cost_max
    lower_half = avg_cost_mid - avg_cost_min

    print("Upper half = ", upper_half)
    print("Lower half = ", lower_half)

    if np.sign(upper_half) == 1 and np.sign(lower_half) == -1:
        new_min = mid_weight
        new_max = max_weight
        new_mid = (new_max - new_min) / 2 + new_min
    elif np.sign(lower_half) == 1 and np.sign(upper_half) == -1:
        new_min = min_weight
        new_max = mid_weight
        new_mid = (new_max - new_min) / 2 + new_min
    elif np.sign(lower_half) == 1 and np.sign(upper_half) == 1 and upper_half < lower_half:
        new_min = mid_weight
        new_max = max_weight
        new_mid = (new_max - new_min) / 2 + new_min
    elif np.sign(lower_half) == 1 and np.sign(upper_half) == 1 and lower_half < upper_half:
        new_min = min_weight
        new_max = mid_weight
        new_mid = (new_max - new_min) / 2 + new_min
    elif np.sign(lower_half) == -1 and np.sign(upper_half) == -1 and upper_half < lower_half:
        new_min = min_weight
        new_max = mid_weight
        new_mid = (new_max - new_min) / 2 + new_min
    elif np.sign(lower_half) == -1 and np.sign(upper_half) == -1 and lower_half < upper_half:
        new_min = mid_weight
        new_max = max_weight
        new_mid = (new_max - new_min) / 2 + new_min

    # if (costs_midpoint["goal_cost"].iloc[-50:]/mid_task.goal_weight_).mean() < max_position_error and (costs_midpoint["orient_cost"].iloc[-50:]/mid_task.goal_orinetation_weight_).mean() < max_orientation_error:
    #     new_min = mid_weight
    #     new_max = max_weight
    #     new_mid = (new_max + new_min) / 2
    # else:
    #     new_min = min_weight
    #     new_max = mid_weight
    #     new_mid = (new_max - new_min) / 2
    
    print("New MIDPOINT = ", new_mid)
    print("Saving...")
    with open("automatic_search_stab_weight_0_120/midpoint.txt", "w") as midpoint_file:
        midpoint_file.write(str(new_mid))
    with open("automatic_search_stab_weight_0_120/min_weight.txt", "w") as min_file:
        min_file.write(str(new_min))
    with open("automatic_search_stab_weight_0_120/max_weight.txt", "w") as max_file:
        max_file.write(str(new_max))
    midpoint_file.close()
    min_file.close()
    max_file.close()
    # np.savetxt(X=new_mid, fname="automatic_search_stab_weight/midpoint.txt")
    # np.savetxt(X=new_min, fname="automatic_search_stab_weight/min_weight.txt")
    # np.savetxt(X=new_max, fname="automatic_search_stab_weight/max_weight.txt")


if __name__ == "__main__":
    main()



