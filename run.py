import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from helper import *
from dubins_controller import *
from dubins_env import *
from a1_controller import *
from a1_env import *
import argparse
import pdb
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from tqdm import tqdm, trange
import json
import time
from eval import evaluate_once
# from a1_learning_hierarchical.motion_imitation.envs.a1_env import A1GymEnv


######################### Parameter STUFF ######################
parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, default=None)
parser.add_argument('--env', type=str, default="car")
parser.add_argument('--horizon', type=int, default=3)
parser.add_argument('--points_per_sec', type=int, default=2) #number of points the neural net adjusts (evenly spaced in time)
parser.add_argument('--trajs', '-t', type=int, default=20)
parser.add_argument('--iterations', type=int, default=200)
parser.add_argument('--lr', type=float, default=4e-5)
parser.add_argument('--dt', type=float, default=0.01)
parser.add_argument('--input_weight', type=float, default=0)

parser.add_argument('--dubins_controller_weights', type=list, default=[3, 3, 3, 3])
parser.add_argument('--dubins_dyn_coeffs', type=list, default=[0.5, 0.25, 0.95]) #friction on v, phi, then scale on inputs

parser.add_argument('--a1_controller_weights', type=list, default=[5, 5, 35, 5, 30])

parser.add_argument('--traj_v_range', type=list, default=[1, 3])
parser.add_argument('--traj_theta_range', type=list, default=[-1, 1])
parser.add_argument('--traj_noise', type=float, default=0)

parser.add_argument('--model_scale', type=float, default=3)

args  = parser.parse_args()
params = vars(args)


####################### LOGGING STUFF #########################
LOG_PATH = "./logs"
BOARD_LOG_PATH = os.path.join(LOG_PATH, "tensorboard_logs")
log = params["run_name"]

if log:
    print("-----------Logging to {} -----------".format(log))
    logdir = os.path.join(LOG_PATH, log)
    os.mkdir(logdir)
    writer = SummaryWriter(log_dir=os.path.join(BOARD_LOG_PATH, log))

################### ENVIRONMENT SELECTION ######################
## Nominal Functions and Cost defined in helper.py
if params["env"] == "car":
    f_nominal = nominals["car"]
    weights = params["dubins_controller_weights"]
    coeffs = params["dubins_dyn_coeffs"]

    controller = Dubins_controller(k_x=weights[0], k_y=weights[1], k_v=weights[2], k_phi=weights[3])
    env = Dubins_env(total_time=params["horizon"], dt=params["dt"], f_v=coeffs[0], f_phi=coeffs[1], scale=coeffs[2])

elif params["env"] == "a1":
    f_nominal = nominals["a1"]
    weights = params["dubins_controller_weights"]

    controller = A1_controller(k_x=weights[0], k_y=weights[1], k_v=weights[2], k_phi=weights[3], k_w=weights[4])
    env = A1GymEnv(total_time=params["horizon"])

else:
    raise NotImplementedError("Environment not implemented")

#################### NEURAL NETWORK SETUP ##########################
# Input: [x0, x1, ..., y0, y1, ..., xd_f, yd_f]
# Output: Deltas on the above
# make_model in helper.py
model = make_model([2*params["horizon"], 64, 64, 2*params["points_per_sec"]*params["horizon"]])

#Times the model affects
output_times = np.linspace(0, params["horizon"], params["points_per_sec"]*params["horizon"] + 1)

#################### TRAINING LOOP ##################################
optimizer = optim.Adam(model.parameters(), lr=params["lr"])

best_loss = np.inf
prog_bar = trange(params["iterations"], leave=True)
for i in prog_bar:

    optimizer.zero_grad() 

    # Collect trajectories (helper.py)
    dynamics, x0s, tasks, points_set = collect_trajs(model, env, controller, params)

    # Construct loss function
    loss = 0
    for (dyn, x0, task, points) in zip(dynamics, x0s, tasks, points_set):
        x = x0

        def f(x,u,t): 
            return f_nominal(x,u,params["dt"]) + dyn[t][2] - f_nominal(dyn[t][0],dyn[t][1],params["dt"]).detach()

        deltas = model(task[:-2])*params["model_scale"]
        task_adj = points + deltas
        spline = Spline(task_adj[:params["horizon"]*params["points_per_sec"]], task_adj[params["horizon"]*params["points_per_sec"]:], times=output_times)

        j = 0
        for t in np.arange(0, params["horizon"], params["dt"]):
            u, des_pos, act_pos= controller.next_action(t, spline, x)

            if (j % 10 == 0):
                loss += cost(x, u, t, task, params)
                
            x = f(x, u, int(t/params["dt"]))

            j += 1

    # Checkpoint
    if log:
        writer.add_scalar("Loss/Train", loss.item(), i)

        if (i % 50 == 0):
            if loss.item() < best_loss:
                torch.save(model, os.path.join(logdir, "model.pt"))
                best_loss = loss.item()


    # Backprop
    loss.retain_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    prog_bar.set_description("Loss: {}".format(loss.item()), refresh=True)


########################### FINAL LOGGING STUFF ###########################
if log:
    writer.close()

    with open(os.path.join(logdir, "params.json"), "w+") as outfile:
        json.dump(params, outfile)



