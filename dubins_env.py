import gym
gym.logger.set_level(40)
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import torch
import pdb


class Dubins_env(gym.Env):
    """
    Description:
        Version of Dubins Car Model
        x, y, v, phi
    """
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, total_time=10, dt=0.01, f_v=0.1, f_phi=0.05, scale=0.95, v0=0, phi0=0):

        max_state = np.array([100, 100, 100, 100])
        max_input = np.array([10, 10])
        self.action_space = spaces.Box(low=-max_input, high=max_input, shape=(2,), dtype=np.float)
        self.observation_space = spaces.Box(low=-max_state, high=max_state, shape=(4,), dtype=np.float)
        
        self.num_steps = total_time // dt
        self.total_time = total_time
        self.dt = dt
        self.curr_step = 0
        self.done = False
        self.f_v = f_v
        self.f_phi = f_phi
        self.scale = scale
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
        a = action[0]
        theta = action[1]

        dot = torch.zeros(4)

        dot[0] = v*torch.cos(phi)
        dot[1] = v*torch.sin(phi)
        dot[2] = -self.f_v*v + self.scale*a
        dot[3] = -self.f_phi*phi + self.scale*theta

        self.state = state_copy + dot*self.dt

        costs = 0

        self.curr_step += 1

        if self.curr_step == self.num_steps:
          self.done=True

        return self._get_obs(), -costs, self.done, {"curr_time": self.curr_step*self.dt}

    def time(self):
        return self.curr_step*self.dt

    #We have this return v_init and phi_init so they can be added to task for input to the model
    def reset(self):

        v_init = np.random.uniform(0, self.v0)
        phi_init = np.random.uniform(-self.phi0, self.phi0)

        self.state = torch.tensor([0, 0, v_init, phi_init], dtype=torch.float)
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
