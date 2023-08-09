import numpy as np 
import matplotlib.pyplot as plt
import os
import sys

def plot_results(res_dir, updates):

    costs = []
    for update in updates[:-2]:
        c=np.loadtxt(res_dir + "/" + update + "/eval_costs.txt")
        costs.append(c)

    costs = np.array(costs)
    cost = costs[:,0]
    cost_stab = costs[:,1]
    cost_goal = costs[:,2]
    cost_orientation = costs[:,3]
    cost_acc = costs[:,4]

    plt.figure(figsize=(20,20))
    plt.plot(cost)
    plt.title("Overall cost")
    plt.grid()
    plt.show()

    plt.figure(figsize=(20,20))
    plt.plot(cost_stab)
    plt.title("Stability cost")
    plt.grid()
    plt.show()

    plt.figure(figsize=(20,20))
    plt.plot(cost_goal)
    plt.title("Goal cost")
    plt.grid()
    plt.show()

    plt.figure(figsize=(20,20))
    plt.plot(cost_orientation)
    plt.title("Orientation cost")
    plt.grid()
    plt.show()

    plt.figure(figsize=(20,20))
    plt.plot(cost_acc)
    plt.title("Acceleration cost")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    res_dir = sys.argv[1]
    updates = np.sort([update for update in os.listdir(res_dir) if "update0" in update])

    plot_results(res_dir, updates)