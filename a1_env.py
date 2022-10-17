import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import torch
import pdb

from a1_learning_hierarchical.motion_imitation.envs.a1_env import A1GymEnv

class A1_env(gym.Env):
    """
    Description:
        Version of Dubins Car Model
    """
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, controller, params, render=False):

        self.env = A1GymEnv(total_time=params["task_time"], dt=params["dt"], render=render)
        self.controller = controller
        self.params = params


    def seed(self,seed=None):
        self.np_random,seed=seeding.np_random(seed)

    def step(self, action):
        obs, cost, done, info = self.env.step(action)

        return obs, cost, done, info

    def time(self):
        return self.curr_step*self.dt

    def reset(self):
        obs = self.env.reset()
        self.env.warm_up = True
        v_des = torch.tensor(np.random.uniform(self.params["a1_warm_up_info"][0], self.params["a1_warm_up_info"][1]), dtype=torch.float)
        phi_des = torch.tensor(0, dtype=torch.float)
        for _ in np.arange(0, self.params["a1_warm_up_info"][2], self.params["dt"]):
            action = self.controller.next_action_warm_up(v_des, phi_des, obs)
            obs, reward, done, info = self.env.step(action)
        self.env.init_time = self.env.robot.GetTimeSinceReset()
        self.env.warm_up = False
        print("--------------WARMED UP----------------")
        return obs

    def render(self):
        return None
