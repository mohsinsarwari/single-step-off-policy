import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
gym.logger.set_level(40)
from helper import *
from task_generator import *
from dubins_controller import *
from dubins_env import *
import argparse
import pdb
import os
import json
from a1_controller import *
from a1_learning_hierarchical.motion_imitation.envs.a1_env import A1GymEnv
matplotlib.rcParams.update({'font.size': 8})

def evaluate(path, model_name, save_fig):

	model = torch.load(os.path.join(path, model_name))
	with open(os.path.join(path, 'params.json')) as json_file:
		params = json.load(json_file)

	#task = figure_eight(radius=params["task_radius"], time=params["task_time"])
	task = random(params["task_time"])

	if params["env"] == "car":
		f_nominal = nominals["car"]

		controller = Dubins_controller(params)
		env = Dubins_env(params)

		num_input_states = 5 # Four states + time

	elif params["env"] == "a1":
		f_nominal = nominals["a1"]

		controller = A1_controller(params)
		env = A1GymEnv(controller, params) #pass in controller for warm up on reset

		num_input_states = 6 # Five states + time

	else:
		raise NotImplementedError("Environment not implemented")

	t = 0
	i = 0
	obs = env.reset()

	tar_xs = []
	tar_ys = []
	act_xs = []
	act_ys = []
	model_xs = []
	model_ys = []

	rollout_len = int(params["model_dt"] / params["dt"])
	num_rollout = int(params["task_time"] / params["model_dt"])
	obs = env.reset()

	for j in range(num_rollout):
		t0 = j * params["model_dt"]
		model_act = model(model_input(obs, t0)).detach() * params["model_scale"]

		x, y = task.evaluate(t0 + params["model_dt"], der=0)
		x_dot, y_dot = task.evaluate(t0 + params["model_dt"], der=1)
		des_pos = [x + model_act[0], y + model_act[1]]
		des_vel = [x_dot + model_act[2], y_dot + model_act[3]]

		model_xs.append(des_pos[0].item())
		model_ys.append(des_pos[1].item())

		for i in range(rollout_len):
			x, y = task.evaluate(t0 + i*params["dt"], der=0)
			u, des_pos, act_pos = controller.next_action(des_pos, des_vel, obs)
			tar_xs.append(x)
			tar_ys.append(y)
			act_xs.append(act_pos[0].item())
			act_ys.append(act_pos[1].item())
			obs, _, _, _ = env.step(u)

	plt.plot(tar_xs, tar_ys, "b", label="task")
	plt.plot(act_xs, act_ys, "orange", label="actual")
	plt.scatter(model_xs, model_ys, label="model_out")
	#head_x, head_y, head_x_dir, head_y_dir = heading_arrays(act[0], act[1], act[2], stride=100)
	#model_ax[i].quiver(head_x, head_y, head_x_dir, head_y_dir, label="heading")
	#model_ax[i].scatter(task_points[:params["horizon"]*params["points_per_sec"]] + x0[0], task_points[params["horizon"]*params["points_per_sec"]:] + x0[1], label="task nodes")
	#model_ax[i].scatter(task_adj[:params["horizon"]*params["points_per_sec"]] + x0[0], task_adj[params["horizon"]*params["points_per_sec"]:] + x0[1], label="model nodes")
	plt.legend()

	# print("Model Loss: ", smart_loss_avg)
	# print("Naive Loss: ", dum_loss_avg)

	if save_fig:
		model_based.savefig(os.path.join(path, "plot.png"),  bbox_inches='tight')
		naive.savefig(os.path.join(path, "plot.png"),  bbox_inches='tight')
	else:
		plt.show()


if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--run_name', '-n', type=str, default="car_test1")
	parser.add_argument('--model_name', type=str, default="best_model.pt")
	parser.add_argument('--save',  action='store_true')
	args  = parser.parse_args()

	evaluate(os.path.join("./logs", args.run_name), args.model_name, args.save)
