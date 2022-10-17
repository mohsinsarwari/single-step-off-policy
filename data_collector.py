"""
@author: Mohsin
"""
from scipy import interpolate
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import pdb

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

		self.x0s = []
		self.t0s = []
		self.actions = []

		self.us = []
		self.all_states = []

	def __iter__(self):
		assert len(self.x0s) > 0, "Must Collect Data First!"
		return self

	def __next__(self):
		if self.index < len(self.x0s):
			return self.rollout_full()

		raise StopIteration


	def rollout_full(self):
		dyn_cuttoff = self.index * int(self.params["model_dt"] / self.params["dt"])
		x0 = self.x0s[self.index]
		t0 = self.t0s[self.index]
		action = self.actions[self.index]
		dyn_tuples = list(zip(self.all_states[dyn_cuttoff:-1], self.us[dyn_cuttoff:], self.all_states[dyn_cuttoff+1:]))

		self.index += 1

		return (x0, t0, action, dyn_tuples)


	def rollout_single(self):
		dyn_cuttoff = self.index * int(self.params["model_dt"] / self.params["dt"])
		dyn_cuttoff_1 = (self.index+1) * int(self.params["model_dt"] / self.params["dt"])
		x0 = self.x0s[self.index]
		t0 = self.t0s[self.index]
		action = self.actions[self.index]
		dyn_tuples = list(zip(self.all_states[dyn_cuttoff:dyn_cuttoff_1], self.us[dyn_cuttoff:dyn_cuttoff_1], self.all_states[dyn_cuttoff+1:dyn_cuttoff_1+1]))

		self.index += 1

		return (x0, t0, action, dyn_tuples)

	def rollout_fixed(self):
		dyn_cuttoff = self.index * int(self.params["model_dt"] / self.params["dt"])
		x0 = self.x0s[self.index]
		t0 = self.t0s[self.index]
		action = self.actions[self.index]
		dyn_tuples = list(zip(self.all_states[dyn_cuttoff:-1], self.us[dyn_cuttoff:], self.all_states[dyn_cuttoff+1:]))

		self.index += 1

		return (x0, t0, action, dyn_tuples)
		
	def collect_data(self, model):
		self.index = 0 # for iteration
		self.x0s = []
		self.t0s = []
		self.actions = []

		self.us = []
		self.all_states = []

		with torch.no_grad():
			rollout_len = int(self.params["model_dt"] / self.params["dt"])
			num_rollout = int(self.params["task_time"] / self.params["model_dt"])
			obs = self.env.reset()
			self.all_states.append(obs)

			for j in range(num_rollout):
				t0 = j * self.params["model_dt"]
				model_act = model(model_input(obs, t0)).detach() * self.params["model_scale"]
				self.x0s.append(obs)
				self.t0s.append(t0)
				self.actions.append(model_act)

				x, y = self.task.evaluate(t0 + self.params["model_dt"], der=0)
				x_dot, y_dot = self.task.evaluate(t0 + self.params["model_dt"], der=1)
				des_pos = [x + model_act[0], y + model_act[1]]
				des_vel = [x_dot + model_act[2], y_dot + model_act[3]]

				for i in range(rollout_len):
					u, des_pos, act_pos = self.controller.next_action(des_pos, des_vel, obs)
					self.us.append(u)
					obs, _, _, _ = self.env.step(u)
					self.all_states.append(obs)

