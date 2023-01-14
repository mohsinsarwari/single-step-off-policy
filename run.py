import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gym
gym.logger.set_level(40)

from tqdm import tqdm, trange
import json
import argparse
import pdb
import os
import shutil

from dubins_controller import *
from dubins_env import *
# from a1_controller import *
# from a1_env import *

from helper import *
from task_generator import *
from data_collector import DataCollector


######################### PARAMETER STUFF ######################
parser = argparse.ArgumentParser()
parser.add_argument('--run_name', '-n', type=str, default=None)
parser.add_argument('--env', type=str, default="car")
parser.add_argument('--iterations', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--input_weight', type=float, default=0) #weight on input in cost function
parser.add_argument('--loss_stride', type=float, default=5) # number of simulation steps before adding cost to loss again
parser.add_argument('--terminal_weight', type=float, default=1)
parser.add_argument('--gamma', type=float, default=0.95)


parser.add_argument('--task_radius', type=float, default=3) #figure eight radius
parser.add_argument('--task_time', type=float, default=10) #time (in seconds) to complete figure eight
parser.add_argument('--model_dt', type=float, default=0.5) #time length between model calls
parser.add_argument('--dt', type=float, default=0.01)
parser.add_argument('--num_model_calls_per_rollout', type=float, default=5)
parser.add_argument('--buffer_length', type=float, default=200)
parser.add_argument('--sim_score_scale', type=float, default=100)

parser.add_argument('--dubins_controller_weights', type=list, default=[3, 3, 3, 3])
parser.add_argument('--dubins_dyn_coeffs', type=list, default=[0.5, 0.25, 0.95, 0, 0]) #friction on v, phi, scale on inputs, init v between [0, v0] and phi between [-phi0, phi0]

parser.add_argument('--a1_controller_weights', type=list, default=[3, 3, 5, 5, 15]) #x, y, v, phi, w, alpha
parser.add_argument('--a1_warm_up_info', type=list, default=[0.3, 0.5, 2])

parser.add_argument('--width', type=float, default=64) #width of hidden layers (x2)
parser.add_argument('--model_scale', type=float, default=1)
parser.add_argument('--save_every', type=float, default=50) #save model every # iterations
parser.add_argument('--overwrite', '-o', action="store_true") #save model every # iterations

parser.add_argument('--critic', action="store_true") # add critic

args  = parser.parse_args()
params = vars(args)


####################### LOGGING STUFF #########################
LOG_PATH = "./logs"
BOARD_LOG_PATH = os.path.join(LOG_PATH, "tensorboard_logs")
log = params["run_name"]

if log:
	print("-----------Logging to {} -----------".format(log))
	logdir = os.path.join(LOG_PATH, log)
	board_logdir = os.path.join(BOARD_LOG_PATH, log)

	if params["overwrite"]:
		if os.path.exists(board_logdir):
			shutil.rmtree(board_logdir)
		if os.path.exists(logdir):
			shutil.rmtree(logdir)

	os.mkdir(logdir)
	writer = SummaryWriter(log_dir=board_logdir)


################### ENVIRONMENT AND TASK SELECTION ######################
## Nominal Functions and Cost defined in helper.py

task = figure_eight(radius=params["task_radius"], time=params["task_time"])

if params["env"] == "car":
	f_nominal = nominals["car"]

	controller = Dubins_controller(params)
	env = Dubins_env(params)

	num_phys_states = 4
	num_input_states = 6 # Four states + 2 time states

elif params["env"] == "a1":
	f_nominal = nominals["a1"]

	controller = A1_controller(params)
	env = A1GymEnv(controller, params) #pass in controller for warm up on reset

	num_phys_states = 5
	num_input_states = 7 # Five states + 2 time states

else:
	raise NotImplementedError("Environment not implemented")

################### MISC ######################
rollout_len = int(params["model_dt"] / params["dt"])
mse_loss = torch.nn.MSELoss()

################### DATA COLLECTOR ######################
collector = DataCollector(task, env, controller, params)

#################### NEURAL NETWORK SETUP ##########################
# Input: state + time
# Output: x_y position and velocity
# make_model in helper.py
model = make_model([num_input_states, params["width"], params["width"], 4])
vfun = make_model([num_phys_states, params["width"], params["width"], 1])

#################### TRAINING LOOP ##################################
optimizer = optim.Adam(model.parameters(), lr=params["lr"])
critic_optimizer = optim.Adam(vfun.parameters(), lr=params["lr"])

best_loss = np.inf
best_iter = 0
prog_bar = trange(params["iterations"], leave=True)
for i in prog_bar:

	optimizer.zero_grad()
	critic_optimizer.zero_grad() 

	################# DATA COLLECTION (see data_collector.py) ################
	collector.collect_data(model)

	######################## LOSS FUNCTION CONSTRUCTION ######################
	rollout = collector.get_next()

	loss = 0
	loss_finite = 0
	critic_loss = 0
	while rollout: 
		x0s = rollout[0]
		t0s = rollout[1]
		actions = rollout[2]
		dyns = rollout[3]

		discount = 1
		initial_obs = None
		rollout_loss = 0
		for j in range(len(x0s)):

			x0 = x0s[j]
			if j == 0:
				initial_obs = x0
			t0 = t0s[j]
			dyn = dyns[j]

			def f(x,u,k):
				return f_nominal(x,u,params["dt"]) - f_nominal(dyn[k][0],dyn[k][1],params["dt"]) + dyn[k][2]
		
			obs = x0

			model_act = model(model_input(obs, t0, params["task_time"])) * params["model_scale"]

			x, y = task.evaluate(t0 + params["model_dt"], der=0)
			x_dot, y_dot = task.evaluate(t0 + params["model_dt"], der=1)
			des_pos = [x + model_act[0], y + model_act[1]]
			des_vel = [x_dot + model_act[2], y_dot + model_act[3]]

			

			for k in range(int(params["model_dt"] / params["dt"])):
				t = t0 + k * params["dt"]
				u, des_pos, act_pos = controller.next_action(des_pos, des_vel, obs)

				# if (k % params["loss_stride"] == 0):
					
				obs_prev, obs = obs, f(obs, u, k)

			stage_cost = discount*cost(obs_prev, u, t, task, params)
			rollout_loss += stage_cost
			loss += stage_cost

			discount *= params["gamma"]
		
		critic_loss += torch.square(vfun(initial_obs) - (loss.detach() + discount * vfun(obs)))
		# print(loss, loss.shape, vfun(obs), vfun(obs).shape)
		loss_finite += rollout_loss
		if params["critic"]:
			loss += discount * vfun(obs).reshape(loss.shape)
		

		rollout = collector.get_next()

	# old_rollout = collector.get_old_next()
	# while old_rollout:

	# 	x0 = old_rollout[0]
	# 	t0 = old_rollout[1]
	# 	action = old_rollout[2]
	# 	dyn = old_rollout[3]

	# 	def f(x,u,k):
	# 		return f_nominal(x,u,params["dt"]) - f_nominal(dyn[k][0],dyn[k][1],params["dt"]) + dyn[k][2]
	
	# 	obs = x0

	# 	model_act = model(model_input(obs, t0, params["task_time"])) * params["model_scale"]

	# 	sim_score = torch.exp(-params["sim_score_scale"] * torch.norm(model_act - action))

	# 	x, y = task.evaluate(t0 + params["model_dt"], der=0)
	# 	x_dot, y_dot = task.evaluate(t0 + params["model_dt"], der=1)
	# 	des_pos = [x + action[0], y + action[1]]
	# 	des_vel = [x_dot + action[2], y_dot + action[3]]

	# 	for k in range(int(params["model_dt"] / params["dt"])):
	# 		t = t0 + k * params["dt"]
	# 		u, des_pos, act_pos = controller.next_action(des_pos, des_vel, obs)

	# 		# if (k % params["loss_stride"] == 0):
				
	# 		obs = f(obs, u, k)

	# 	old_loss = (sim_score * cost(obs, u, t, task, params))

	# 	old_rollout = collector.get_next()

	# loss = loss + old_loss


	# Checkpoint
	loss_avg = loss.item()
	critic_loss_avg = critic_loss.item()
	loss_finite_avg = loss_finite.item()
	if log: 
		writer.add_scalar("Loss/Train", loss_avg, i)
		writer.add_scalar("Loss/VFun", critic_loss_avg, i)
		writer.add_scalar("Loss/Finite", loss_finite_avg, i)

		if (i % params["save_every"] == 0) and (i != 0):
			print("Saved Model with Loss {}".format(loss_avg))
			torch.save(model, os.path.join(logdir, "model_{}.pt".format(i)))
			if loss_avg < best_loss:
				print("New best!")
				best_iter = i
				torch.save(model, os.path.join(logdir, "best_model.pt"))
				best_loss = loss_avg


	# Backprop
	loss.retain_grad()
	critic_loss.retain_grad()
	loss.backward(retain_graph=True)
	critic_loss.backward(retain_graph=True)
	optimizer.step()
	critic_optimizer.step()

	prog_bar.set_description(f"Loss/Finite/Critic: {loss_avg}/{loss_finite_avg}/{critic_loss_avg}", refresh=True)


########################### FINAL LOGGING STUFF ###########################
if log:
	writer.close()
	if loss_avg < best_loss:
		best_iter = i
		torch.save(model, os.path.join(logdir, "best_model.pt"))
	torch.save(model, os.path.join(logdir, "final_model.pt"))
	with open(os.path.join(logdir, "params.json"), "w+") as outfile:
		json.dump(params, outfile)



