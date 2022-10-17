import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import torch
import pdb


class A1_env(gym.Env):
    """
    Description:
        Version of Dubins Car Model
    """
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, total_time=10, dt=0.01, v0=0, phi0=0):

        max_state = np.array([100, 100, 100, 100, 100])
        max_input = np.array([10, 10])
        self.action_space = spaces.Box(low=-max_input, high=max_input, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-max_state, high=max_state, shape=(5,), dtype=np.float32)
        
        self.num_steps = total_time // dt
        self.total_time = total_time
        self.dt = dt
        self.curr_step = 0
        self.done = False
        self.v0 = v0
        self.phi0 = phi0

    def seed(self,seed=None):
        self.np_random,seed=seeding.np_random(seed)

    def step(self, action):
        state_copy = self.state.clone()
        x = state_copy[0]
        y = state_copy[1]
        v = state_copy[2]
        phi = state_copy[3]
        w = state_copy[4]
        a = action[2]#.clone()
        theta = action[3]#.clone()

        dot = torch.zeros(5)

        dot[0] = v*torch.cos(phi)
        dot[1] = v*torch.sin(phi)
        dot[2] = a
        dot[3] = w
        dot[4] = theta

        self.state = state_copy + dot*self.dt

        costs = 0

        self.curr_step += 1

        if self.curr_step == self.num_steps:
          self.done=True

        return self._get_obs(), -costs, self.done, {"curr_time": self.curr_step*self.dt}

    def time(self):
        return self.curr_step*self.dt

    def reset(self):
        v_init = np.random.uniform(0, self.v0)
        phi_init = np.random.uniform(-self.phi0, self.phi0)

        self.state = torch.tensor([0, 0, v_init, phi_init, 0], dtype=torch.float)
        self.curr_step = 0
        self.done = False

        return self._get_obs()

    def _get_obs(self):
        return self.state

    def render(self):
        return None

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def angle_normalize(self, x):
        return abs(((x + np.pi) % (2 * np.pi)) - np.pi)
