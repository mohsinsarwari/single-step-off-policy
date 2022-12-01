"""
@author: Mohsin
"""
from scipy import interpolate
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import pdb
import itertools

from helper import *

class DataCollector:
	"""
	Data Collecting Class
	@params:
		task - Spline object defining task
		env - environment to run
		controller - controller for environment
		params - parameters for run
	"""
	def __init__(self, task, env, controller, params):
		self.env = env
		self.controller = controller
		self.task = task
		self.params = params

		self.index = 0
		self.index_old = 0

		self.x0s_buffer = []
		self.t0s_buffer = []
		self.actions_buffer = []
		self.states_action_tuples_buffer = []

		self.x0s = []
		self.t0s = []
		self.actions = []
		self.states_action_tuples = []

	def get_next(self):
		num = self.params["num_model_calls_per_rollout"]
		if self.index + num <= len(self.x0s):
			return self.rollout_fixed()
		return None

	def get_old_next(self):
		if self.index < len(self.x0s_buffer):
			return self.rollout_old()
		return None

	def rollout_fixed(self):
		num = self.params["num_model_calls_per_rollout"]

		r_x0s = self.x0s[self.index:self.index+num]
		r_t0s = self.t0s[self.index:self.index+num]
		r_actions = self.actions[self.index:self.index+num]
		dyn_tuples_list = self.states_action_tuples[self.index:self.index+num]

		self.index += 1

		return (r_x0s, r_t0s, r_actions, dyn_tuples_list)


	def rollout_old(self):

		r_x0 = self.x0s_buffer[self.index_old]
		r_t0 = self.t0s_buffer[self.index_old]
		r_action = self.actions_buffer[self.index_old]
		dyn_tuples_list = self.states_action_tuples_buffer[self.index_old]

		self.index_old += 1

		return (r_x0, r_t0, r_action, dyn_tuples_list)
		
	def collect_data(self, model):
		self.index = 0 # for iteration
		self.index_old = 0 # for iteration

		self.x0s = []
		self.t0s = []
		self.actions = []
		self.states_action_tuples = []

		with torch.no_grad():
			rollout_len = int(self.params["model_dt"] / self.params["dt"])
			num_rollout = int(self.params["task_time"] / self.params["model_dt"])
			obs = self.env.reset()

			for j in range(num_rollout):
				t0 = j * self.params["model_dt"]
				model_act = model(model_input(obs, t0, self.params["task_time"])).detach() * self.params["model_scale"]

				self.x0s.append(obs)
				self.t0s.append(t0)
				self.actions.append(model_act)
				self.states_action_tuples.append([])

				if len(self.x0s_buffer) > self.params["buffer_length"]:
					self.x0s_buffer.pop(0)
					self.t0s_buffer.pop(0)
					self.actions_buffer.pop(0)
					self.states_action_tuples_buffer.pop(0)

				self.x0s_buffer.append(obs)
				self.t0s_buffer.append(t0)
				self.actions_buffer.append(model_act)
				self.states_action_tuples_buffer.append([])


				x, y = self.task.evaluate(t0 + self.params["model_dt"], der=0)
				x_dot, y_dot = self.task.evaluate(t0 + self.params["model_dt"], der=1)
				des_pos = [x + model_act[0], y + model_act[1]]
				des_vel = [x_dot + model_act[2], y_dot + model_act[3]]

				for i in range(rollout_len):
					u, des_pos, act_pos = self.controller.next_action(des_pos, des_vel, obs)
					next_obs, _, _, _ = self.env.step(u)
					self.states_action_tuples[j].append((obs, u, next_obs))
					self.states_action_tuples_buffer[-1].append((obs, u, next_obs))
					obs = next_obs

