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
import os
import json



def evaluate_once(path):

	model = torch.load(os.path.join(path, "model.pt"))

	with open(os.path.join(path, 'params.json')) as json_file:
		params = json.load(json_file)

	if params["env"] == "car":
		controller = Dubins_controller(k_x=3, k_y=3, k_v=3, k_phi=3)
		env = Dubins_env(total_time=params["horizon"], dt=params["dt"])
		dum_env = Dubins_env(total_time=params["horizon"], dt=params["dt"])

	task, xd_f, yd_f = generate_traj(params["horizon"])
	spline_params = model(task)*params["model_scale"]

	spline = Spline(spline_params[:params["horizon"]], spline_params[params["horizon"]:], xd_f=xd_f, yd_f=yd_f)
	task_spline = Spline(task[:params["horizon"]], task[params["horizon"]:], xd_f=xd_f, yd_f=yd_f)

	des_x = []
	des_y = []
	act_x = []
	act_y = []
	tar_x = []
	tar_y = []

	#naive follower
	dum_x = []
	dum_y = []

	obs = env.reset()
	dum_obs = dum_env.reset()
	smart_loss = 0
	dum_loss = 0
	for t in np.arange(0, params["horizon"], params["dt"]):
		u, des_pos, act_pos= controller.next_action(t, spline, obs)
		dum_u, _, dum_pos = controller.next_action(t, task_spline, dum_obs)

		obs, reward, done, info = env.step(u)
		dum_obs, _, _, _ = dum_env.step(dum_u)

		target = Spline(task[:params["horizon"]], task[params["horizon"]:], xd_f=xd_f, yd_f=yd_f)
		tar_pos_x, tar_pos_y = target.evaluate(t, der=0)

		des_x.append(des_pos[0].detach().numpy())
		des_y.append(des_pos[1].detach().numpy())
		act_x.append(act_pos[0].detach().numpy())
		act_y.append(act_pos[1].detach().numpy())
		tar_x.append(tar_pos_x.detach().numpy())
		tar_y.append(tar_pos_y.detach().numpy())
		dum_x.append(dum_pos[0].detach().numpy())
		dum_y.append(dum_pos[1].detach().numpy())

		smart_loss += cost(obs, u, t, task, xd_f, yd_f, params).detach().numpy()
		dum_loss += cost(dum_obs, dum_u, t, task, xd_f, yd_f, params).detach().numpy()

	return [des_x, des_y], [act_x, act_y], [tar_x, tar_y], [dum_x, dum_y], [smart_loss, dum_loss]

def cost(x, u, t, task, xd_f, yd_f, params):

    spline = Spline(task[:params["horizon"]], task[params["horizon"]:], xd_f=xd_f, yd_f=yd_f)
    x_d, y_d = spline.evaluate(t, der=0)

    return ((x[0] - x_d)**2 + (x[1] - y_d)**2) + (params["input_weight"] * (u[0]**2 + u[1]**2))

def evaluate(path):

	trials = 3

	fig, ax = plt.subplots(trials, 2)

	smart_loss_avg = 0
	dum_loss_avg = 0

	for i in range(trials):
		des, act, tar, dum, loss = evaluate_once(path)
		smart_loss_avg += loss[0] / trials
		dum_loss_avg += loss[1] / trials

		ax[i, 0].set_title("Using Model {} (Loss: {})".format(i, loss[0]))
		ax[i, 0].plot(des[0], des[1], label="model output")
		ax[i, 0].plot(act[0], act[1], label="actual")
		ax[i, 0].plot(tar[0], tar[1], label="task")
		ax[i, 0].legend()

		ax[i, 1].set_title("Naive {} (Loss: {})".format(i, loss[1]))
		ax[i, 1].plot(dum[0], dum[1], label="actual")
		ax[i, 1].plot(tar[0], tar[1], label="task")
		ax[i, 1].legend()

	for a in ax.flat:
	    a.label_outer()

	print("Avg Model Loss: ", smart_loss_avg)
	print("Avg Naive Loss: ", dum_loss_avg)

	plt.show()


if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--run_name', type=str, default="car_test1")
	args  = parser.parse_args()

	evaluate(os.path.join("./logs", args.run_name))
