import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
gym.logger.set_level(40)
from helper import *
from dubins_controller import *
from dubins_env import *
import argparse
import pdb
import os
import json
from a1_controller import *
from a1_learning_hierarchical.motion_imitation.envs.a1_env import A1GymEnv
matplotlib.rcParams.update({'font.size': 8})



def evaluate_once(model, params, v=None, theta=None):

	if params["env"] == "car":
		weights = params["dubins_controller_weights"]
		coeffs = params["dubins_dyn_coeffs"]

		controller = Dubins_controller(k_x=weights[0], k_y=weights[1], k_v=weights[2], k_phi=weights[3])
		env = Dubins_env(total_time=params["horizon"], dt=params["dt"], f_v=coeffs[0], f_phi=coeffs[1], scale=coeffs[2], v0=coeffs[3], phi0=coeffs[4])
		dum_env = Dubins_env(total_time=params["horizon"], dt=params["dt"], f_v=coeffs[0], f_phi=coeffs[1], scale=coeffs[2], v0=coeffs[3], phi0=coeffs[4])

	elif params["env"] == "a1":
		weights = params["a1_controller_weights"]

		controller = A1_controller(k_x=weights[0], k_y=weights[1], k_v=weights[2], k_phi=weights[3], k_w=weights[4])
		env = A1GymEnv(total_time=params["horizon"], dt=params["dt"])
		dum_env = A1GymEnv(total_time=params["horizon"], dt=params["dt"])
		# env = A1_env(total_time=params["horizon"], dt=params["dt"])
		# dum_env = A1_env(total_time=params["horizon"], dt=params["dt"])
	
	output_times = np.linspace(0, params["horizon"], params["points_per_sec"]*params["horizon"] + 1)

	if v==None:
		v = params["traj_v_range"]
	else:
		v = [v, v]

	if theta==None:
		theta = params["traj_theta_range"]
	else:
		theta = [theta, theta]

	if params["env"] == "car":
		obs = env.reset()
		dum_obs = dum_env.reset()
	elif params["env"] == "a1":
		obs = a1_warm_up(env, controller, params)
		dum_obs = a1_warm_up(dum_env, controller, params)

	x0 = obs
	dum_x0 = dum_obs

	task = generate_traj(params["horizon"], params["traj_noise"], v, theta)
	task_points = find_points(task, params)

	deltas = model(model_input(task, obs, params))*params["model_scale"]
	task_adj = task_points + deltas
	spline = Spline(task_adj[:params["horizon"]*params["points_per_sec"]], task_adj[params["horizon"]*params["points_per_sec"]:], times=output_times, init_pos=x0)
	task_spline = Spline(task[:params["horizon"]], task[params["horizon"]:], init_pos=x0)
	dum_task_spline = Spline(task[:params["horizon"]], task[params["horizon"]:], init_pos=dum_obs)

	des_x = []
	des_y = []
	act_x = []
	act_y = []
	act_phi = []
	tar_x = []
	tar_y = []
	#naive follower
	dum_x = []
	dum_y = []
	dum_phi = []
	dum_tar_x = []
	dum_tar_y = []


	smart_loss = 0
	dum_loss = 0
	i = 0
	for t in np.arange(0, params["horizon"] + params["dt"], params["dt"]):
		if (i % params["controller_stride"] == 0):
			u, des_pos, act_pos= controller.next_action(t, spline, obs)
			dum_u, _, dum_pos = controller.next_action(t, dum_task_spline, dum_obs)
			des_x.append(des_pos[0].detach().item())
			des_y.append(des_pos[1].detach().item())
			tar_pos_x, tar_pos_y = task_spline.evaluate(t, der=0)
			dum_tar_pos_x, dum_tar_pos_y = dum_task_spline.evaluate(t, der=0)

		obs, reward, done, info = env.step(u)
		dum_obs, _, _, _ = dum_env.step(dum_u)

		act_x.append(obs[0].detach().item())
		act_y.append(obs[1].detach().item())
		act_phi.append(obs[3].detach().item())
		tar_x.append(tar_pos_x.detach().item())
		tar_y.append(tar_pos_y.detach().item())
		dum_x.append(dum_obs[0].detach().item())
		dum_y.append(dum_obs[1].detach().item())
		dum_phi.append(dum_obs[3].detach().item())
		dum_tar_x.append(dum_tar_pos_x.detach().item())
		dum_tar_y.append(dum_tar_pos_y.detach().item())

		smart_loss += cost(obs, u, t, task, params, x0).detach().item()
		dum_loss += cost(dum_obs, dum_u, t, task, params, dum_x0).detach().item()

		i += 1

	return [des_x, des_y], [act_x, act_y, act_phi], [tar_x, tar_y], [dum_x, dum_y, dum_phi], [dum_tar_x, dum_tar_y],[smart_loss, dum_loss], task_points, task_adj.detach(), x0, dum_x0

def evaluate(path, model_name, save_fig):

	trials = 2

	fig, ax = plt.subplots(trials, 2, sharex="row", sharey="row", figsize=(12, 9))

	smart_loss_avg = 0
	dum_loss_avg = 0

	model = torch.load(os.path.join(path, model_name))
	with open(os.path.join(path, 'params.json')) as json_file:
		params = json.load(json_file)

	eval_values = [[0.4, -0.3], [0.4, 0.3], [0.4, 0.01]]

	for i in range(trials):
		v, theta = eval_values[i]
		des, act, tar, dum, dum_tar, loss, task_points, task_adj, x0, dum_x0 = evaluate_once(model, params, v, theta)
		smart_loss_avg += loss[0] / trials
		dum_loss_avg += loss[1] / trials

		ax[i, 0].set_title("Trial {}: With Model (Loss: {})".format(i, np.round(loss[0], 2)))
		ax[i, 0].plot(tar[0], tar[1], "b", label="task")
		ax[i, 0].plot(des[0], des[1], label="model output")
		ax[i, 0].plot(act[0], act[1], "orange", label="actual")
		head_x, head_y, head_x_dir, head_y_dir = heading_arrays(act[0], act[1], act[2], stride=500)
		ax[i, 0].quiver(head_x, head_y, head_x_dir, head_y_dir, label="heading")
		ax[i, 0].scatter(task_points[:params["horizon"]*params["points_per_sec"]] + x0[0], task_points[params["horizon"]*params["points_per_sec"]:] + x0[1], label="task nodes")
		ax[i, 0].scatter(task_adj[:params["horizon"]*params["points_per_sec"]] + x0[0], task_adj[params["horizon"]*params["points_per_sec"]:] + x0[1], label="model nodes")
		ax[i, 0].legend()

		ax[i, 1].set_title("Trial {}: Naive (Loss: {})".format(i, np.round(loss[1], 2)))
		ax[i, 1].plot(dum_tar[0], dum_tar[1], "b", label="task")
		ax[i, 1].plot(dum[0], dum[1], "orange", label="actual")
		dum_head_x, dum_head_y, dum_head_x_dir, dum_head_y_dir = heading_arrays(dum[0], dum[1], dum[2], stride=500)
		ax[i, 1].quiver(dum_head_x, dum_head_y, dum_head_x_dir, dum_head_y_dir, label="heading")
		ax[i, 1].legend()

	print("Avg Model Loss: ", smart_loss_avg)
	print("Avg Naive Loss: ", dum_loss_avg)

	if save_fig:
		plt.savefig(os.path.join(path, "plot.png"),  bbox_inches='tight')
	else:
		plt.show()


if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--run_name', '-n', type=str, default="car_test1")
	parser.add_argument('--model_name', type=str, default="best_model.pt")
	parser.add_argument('--save',  action='store_true')
	args  = parser.parse_args()

	evaluate(os.path.join("./logs", args.run_name), args.model_name, args.save)
