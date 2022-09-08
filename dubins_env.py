import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path


class Dubins_env(gym.Env):
    """
    Description:
        Version of Dubins Car Model
    """
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, total_time = 10, dt= 0.01):

        max_state = np.array([100, 100, 100, 100])
        max_input = np.array([10, 10])
        self.action_space = spaces.Box(low=-max_input, high=max_input, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-max_state, high=max_state, shape=(4,), dtype=np.float32)
        self.state = np.array([0, 0, 0, 0])
        
        self.num_steps = total_time // dt
        self.total_time = total_time
        self.dt = dt
        self.curr_step = 0
        self.done = False

    def seed(self,seed=None):
        self.np_random,seed=seeding.np_random(seed)

    def step(self, action):
        x, y, v, phi = self.state
        a, theta = action

        x_dot = v*np.cos(phi)
        y_dot = v*np.sin(phi)
        v_dot = a
        phi_dot = theta

        x_new = x + x_dot*self.dt
        y_new = y + y_dot*self.dt
        v_new = v + v_dot*self.dt
        phi_new = phi + phi_dot*self.dt


        self.state = np.array([x_new, y_new, v_new, phi_new])

        costs = 0

        self.curr_step += 1

        if self.curr_step == self.num_steps:
          self.done=True

        return self._get_obs(), -costs, self.done, {"curr_time": self.curr_step*self.dt}

    def time(self):
        return self.curr_step*self.dt

    def reset(self):

        self.state = np.array([0, 0, 0, 0])
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
