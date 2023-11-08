import gymnasium as gym
import numpy as np
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import os
import subprocess

class MetaLearningEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=0, high=60,
                                            shape=(1, 1), dtype=np.uint8)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=10,
                                            shape=(1, 4), dtype=np.uint8)

    def step(self, action):
        # Save action as stability weight
        print("Apply action...")
        print(action)
        with open("meta_learning_stability_weight/weight.txt", "w") as weight_file:
            weight_file.write(str(action[0][0]))
        
        print("Run learning...")
        # Run the main learning for 30 updates
        # process = subprocess.run("/home/hdd/ksavevska/dmpbbo/examples/catkin_ws/src/reaching_task/run_one_learning.bash", capture_output=True)
        # process.wait()
        os.system("bash run_one_learning.bash")

        print("Collect observations...")
        # calculate costs as observations
        observation = self.calculate_costs("meta_learning_stability_weight/results_w_stab_"+str(action[0][0])+"/", action[0][0])
        print(observation)

        print("Calculate reward...")
        # calculate reward from observations
        reward = self.calculate_reward(observation)
        print(reward)
        
        terminated = False
        truncated = False
        info = dict()
        return observation, reward, terminated, truncated, info
    
    def calculate_costs(self, results_folder, stability_weight):
        updates = np.sort(os.listdir(results_folder))
        costs = []
        for update in updates:
            if "update0" in update and update!="update00030":
                files = os.listdir(os.path.join(os.getcwd(), results_folder + update))
                for f in files:
                    if "eval_costs.txt" in f:
                        c = np.loadtxt(os.path.join(os.getcwd(), results_folder + update) + "/" + f)
                        costs.append(c)
        costs = np.array(costs)
        avg_costs = np.zeros((1, costs.shape[1]))
        avg_costs[0][0] = np.average((costs[:,1]/stability_weight) + (costs[:,2]/20.0) + (costs[:, 3]/4.0))
        avg_costs[0][1] = np.average((costs[:,1]/stability_weight)) 
        avg_costs[0][2] = np.average((costs[:,2]/20.0)) 
        avg_costs[0][3] = np.average((costs[:, 3]/4.0)) 

        return avg_costs

    def calculate_reward(self, obs):
        reward = 1/(1 + obs[0][0])
        # reward = np.exp(-obs[0][1])
        return reward
    
    def reset(self, seed=None, options=None):
        observation = 10*np.ones((1,4))
        return observation, ""


def main():
    # Custom environment
    env = MetaLearningEnv()

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("ppo_meta_learning")

    # del model # remove to demonstrate saving and loading

    # model = PPO.load("ppo_cartpole")

    # obs = env.reset()
    # while True:
    #     action, _states = model.predict(obs)
    #     obs, reward = env.step(action)
    #     print("Action = ", action)
    #     print("Costs = ", obs)
    #     print("Action = ", action)


if __name__ == "__main__":
    main()