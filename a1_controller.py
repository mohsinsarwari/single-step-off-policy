import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from a1_env import A1_env
from spline import Spline
import matplotlib
import matplotlib.pyplot as plt
import torch
import pdb
from helper import *
from a1_learning_hierarchical.motion_imitation.envs.a1_env import A1GymEnv

class A1_controller:
    """
    Description:
        Controller for Dubins_env
    """
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, k_x=1, k_y=1, k_v=1, k_phi=1, k_w=1):

        #position gain
        self.k_x = k_x
        self.k_y = k_y

        #speed gain
        self.k_v = k_v

        #angle gain
        self.k_phi = k_phi

        #anglular speed gain 
        self.k_w = k_w

        # self.phi_hat = 0
        # self.alpha = alpha
        # self.count = 0
    """
    @params
        curr_time: time at which to evaluate controls
        path: spline object

    @return
        action: [a, theta] action to apply
        x_d, y_d: desired location
        x_act, y_act: actual location
    """
    def next_action(self, curr_time, spline, obs):

        x_d, y_d = spline.evaluate(curr_time, der=0)
        x_dot_d, y_dot_d = spline.evaluate(curr_time, der=1)

        x_act = obs[0]
        y_act = obs[1]
        v_act = obs[2]
        phi_act = obs[3]
        w_act = obs[4]

        x_tilde_dot_d = x_dot_d + self.k_x*(x_d - x_act)
        y_tilde_dot_d = y_dot_d + self.k_y*(y_d - y_act)

        v_des = torch.sqrt(x_tilde_dot_d**2 + y_tilde_dot_d**2 + 1e-8) #small add needed to make sure backward pass works
        #https://discuss.pytorch.org/t/runtimeerror-function-sqrtbackward-returned-nan-values-in-its-0th-output/48702

        if x_tilde_dot_d == 0:
            phi_des = torch.sign(y_tilde_dot_d)*np.pi/2
        else:
            phi_des = torch.arctan(y_tilde_dot_d/x_tilde_dot_d)

        #change to measure from positive x-axis in the range of 0 to 2pi
        if x_tilde_dot_d < 0:
            phi_des += np.pi
        elif y_tilde_dot_d < 0:
            phi_des += 2*np.pi


        if torch.abs(phi_des - phi_act) > torch.abs(phi_des - 2*np.pi - phi_act):
            phi_des -= 2*np.pi

        # if True or (curr_time == 0):
        #     self.phi_hat = phi_des
        # else:
        #     if (self.count % 100 == 0):
        #         self.phi_hat = (self.alpha*self.phi_hat + phi_des) / (1 + self.alpha)
            
        a = self.k_v*(v_des - v_act)

        w_tilde = self.k_phi*(phi_des - phi_act)

        theta = self.k_w*(w_tilde - w_act)

        action = [v_des, w_tilde, a, theta]

        # self.count += 1

        return action, [x_d, y_d], [x_act, y_act, phi_act]


    def next_action_warm_up(self, v_des, phi_des, obs):

        # x_d, y_d = spline.evaluate(curr_time, der=0)
        # x_dot_d, y_dot_d = spline.evaluate(curr_time, der=1)

        x_act = obs[0]
        y_act = obs[1]
        v_act = obs[2]
        phi_act = obs[3]
        w_act = obs[4]

        a = self.k_v*(v_des - v_act)

        w_tilde = self.k_phi*(phi_des - phi_act)

        theta = self.k_w*(w_tilde - w_act)

        action = [v_des, w_tilde, a, theta]

        return action


if __name__=="__main__":

    horizon = 5
    dt = 0.002
    controller_stride = 25
    
    env = A1GymEnv(total_time=horizon, dt=dt)
    #env = A1_env(total_time=horizon, dt=dt)
                            #  x  y  v  phi w
    controller = A1_controller(3, 3, 5, 5, 15)

    #controller = A1_controller(1, 1, 1, 1, 1)


    params = generate_traj(horizon, 0, [0.4, 0.4], [0.4, 0.4])

    #params = generate_traj(horizon, 0, [0.4, 0.4], [0.2, 0.2])

    #obs = env.reset()

    obs = a1_warm_up(env, controller, {"a1_warm_up_time": 1, "a1_warm_up_vel": [0.4, 0.4], "dt": dt})

    cs = Spline(params[:horizon], params[horizon:], init_pos=obs)

    done = False
    curr_time = 0

    des_x = []
    des_y = []
    act_x = []
    act_y = []
    act_phi = []
    i = 0
    while not done:
        if (i % controller_stride == 0):
            action, des_pos, act_pos = controller.next_action(curr_time, cs, obs)
            des_x.append(des_pos[0])
            des_y.append(des_pos[1])

        obs, _, done, info = env.step(action)
        curr_time = info["curr_time"]

        act_x.append(obs[0])
        act_y.append(obs[1])
        act_phi.append(obs[2])

        i += 1

    head_x, head_y, head_x_dir, head_y_dir = heading_arrays(act_x, act_y, act_phi, stride=500)

    fig, ax = plt.subplots(1, 2)

    ax[0].plot(des_x, des_y, label="desired")
    ax[0].plot(act_x, act_y, label="actual")
    ax[0].quiver(head_x, head_y, head_x_dir, head_y_dir, label="heading")
    # plt.xlim([-3, 3])
    # plt.ylim([-3, 3])
    ax[0].legend()

    #ax[1].plot(np.subtract(des_x, act_x)**2 + np.subtract(des_y, act_y)**2, label="tracking_error")
    ax[1].plot(act_phi, label="phi")
    ax[1].legend()
    plt.show()


