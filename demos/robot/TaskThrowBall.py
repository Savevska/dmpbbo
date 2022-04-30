import numpy as np
from matplotlib import pyplot as plt

from dmpbbo.bbo_for_dmps.Task import Task


class TaskThrowBall(Task):
    def __init__(self, x_goal, x_margin, y_floor, acceleration_weight=0.0001):

        self.x_goal = x_goal
        self.x_margin = x_margin
        self.y_floor = y_floor
        self.acceleration_weight = acceleration_weight

    def cost_labels(self):
        return ["landing site", "acceleration"]

    def evaluate_rollout(self, cost_vars, sample):
        n_dims = 2
        n_time_steps = cost_vars.shape[0]

        # ts = cost_vars[:,0]
        # y = cost_vars[:,1:1+n_dims]
        ydd = cost_vars[:, 1 + n_dims * 2 : 1 + n_dims * 3]
        ball = cost_vars[:, -2:]
        ball_final_x = ball[-1, 0]

        dist_to_landing_site = abs(ball_final_x - self.x_goal)
        dist_to_landing_site -= self.x_margin
        if dist_to_landing_site < 0.0:
            dist_to_landing_site = 0.0

        sum_ydd = 0.0
        if self.acceleration_weight > 0.0:
            sum_ydd = np.sum(np.square(ydd))

        costs = np.zeros(1 + 2)
        costs[1] = dist_to_landing_site
        costs[2] = self.acceleration_weight * sum_ydd / n_time_steps
        costs[0] = np.sum(costs[1:])
        return costs

    def plot_rollout(self, cost_vars, ax=None):
        """Plot y of DMP trajectory"""
        if not ax:
            ax = plt.axes()
        # t = cost_vars[:, 0]
        y = cost_vars[:, 1:3]
        ball = cost_vars[:, -2:]

        line_handles = ax.plot(y[:, 0], y[:, 1], linewidth=0.5)
        line_handles_ball_traj = ax.plot(ball[:, 0], ball[:, 1], "-")
        # line_handles_ball = ax.plot(ball[::5,0],ball[::5,1],'ok')
        # plt.setp(line_handles_ball,'MarkerFaceColor','none')

        line_handles.extend(line_handles_ball_traj)

        # Plot the floor
        xg = self.x_goal
        x_floor = [-1.0, xg - self.x_margin, xg, xg + self.x_margin, 0.4]
        yf = self.y_floor
        y_floor = [yf, yf, yf - 0.05, yf, yf]
        ax.plot(x_floor, y_floor, "-k", linewidth=1)
        ax.plot(self.x_goal, self.y_floor - 0.05, "og")
        ax.axis("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim([-0.9, 0.3])
        ax.set_ylim([-0.4, 0.3])

        return line_handles, ax
