import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from dubins_env import Dubins_env
from spline import Spline
import matplotlib
import matplotlib.pyplot as plt
import torch
import pdb
from helper import *
from task_generator import *

class Dubins_controller:
	"""
	Description:
		Controller for Dubins_env
	"""
	metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

	def __init__(self, params):

		#position gain
		self.k_x = params["dubins_controller_weights"][0]
		self.k_y = params["dubins_controller_weights"][1]

		#speed gain
		self.k_v = params["dubins_controller_weights"][2]

		#angle gain
		self.k_phi = params["dubins_controller_weights"][3]

	"""
	@params
        des_pos: target position
        des_vel: target velocity
        obs: current state

	@return
		action: [a, theta] action to apply
		x_d, y_d: desired location
		x_act, y_act: actual location
	"""
	def next_action(self, des_pos, des_vel, obs):

		x_d, y_d = des_pos
		x_dot_d, y_dot_d = des_vel

		x_act = obs[0]
		y_act = obs[1]
		v_act = obs[2]
		phi_act = obs[3]

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
			
		a = self.k_v*(v_des - v_act)
		theta = self.k_phi*(phi_des - phi_act)

		action = [a, theta]

		return action, [x_d, y_d], [x_act, y_act, phi_act]

if __name__=="__main__":
	params = {"task_time": 4, "dubins_dyn_coeffs": [0.5, 0.25, 0.95, 0, 0], "dubins_controller_weights": [3, 3, 3, 3], "dt": 0.01}
	env = Dubins_env(params)
	controller = Dubins_controller(params)

	task = figure_eight(radius=4, time=params["task_time"])
	#task = random(time=params["task_time"])

	obs = env.reset()
	print(obs)
	done = False
	curr_time = 0

	des_x = []
	des_y = []
	act_x = []
	act_y = []
	act_phi = []

	while not done:
		x, y = task.evaluate(curr_time, der=0)
		x_d, y_d = task.evaluate(curr_time, der=1)
		action, des_pos, act_pos = controller.next_action([x, y], [x_d, y_d], obs)
		obs, _, done, info = env.step(action)
		curr_time = info["curr_time"]

		des_x.append(des_pos[0])
		des_y.append(des_pos[1])
		act_x.append(act_pos[0])
		act_y.append(act_pos[1])
		act_phi.append(act_pos[2])

	head_x, head_y, head_x_dir, head_y_dir = heading_arrays(act_x, act_y, act_phi, stride=100)
	plt.quiver(head_x, head_y, head_x_dir, head_y_dir, label="heading")
	plt.plot(des_x, des_y, label="desired")
	plt.plot(act_x, act_y, label="actual")
	# plt.xlim([-3, 3])
	# plt.ylim([-3, 3])
	plt.legend()
	plt.show()


