import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from helper import *
from dubins_controller import *
from dubins_env import *
import argparse
import pdb

def evaluate(model_path):
	model = torch.load(model_path)

    spline_params = model(task)*model_scale

    spline = Spline(times, spline_params[:total_time], spline_params[total_time:])

    des_x = []
    des_y = []
    act_x = []
    act_y = []
    tar_x = []
    tar_y = []

	for t in np.arange(0, total_time, dt):
        u, des_pos, act_pos= controller.next_action(t, spline, x)

        target = Spline(times, task[1:total_time+1], task[total_time+2:], task[0], task[total_time+1])
        tar_pos_x, tar_pos_y = target.evaluate(t, der=0)

        des_x.append(des_pos[0])
        des_y.append(des_pos[1])

	plt.plot(des_x, des_y, label="model output")
	plt.plot(act_x, act_y, label="actual")
	plt.plot(tar_x, tar_y, label="desired")
	plt.legend()
	plt.show()