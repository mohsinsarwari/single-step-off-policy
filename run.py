import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
gym.logger.set_level(40)
from helper import *
from dubins_controller import *
from dubins_env import *
from a1_controller import *
from a1_env import *
import argparse
import pdb
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm, trange
import json
import time
import shutil
from a1_learning_hierarchical.motion_imitation.envs.a1_env import A1GymEnv


######################### PARAMETER STUFF ######################
parser = argparse.ArgumentParser()
parser.add_argument('--run_name', '-n', type=str, default=None)
parser.add_argument('--env', type=str, default="a1")
parser.add_argument('--horizon', type=int, default=5)
parser.add_argument('--points_per_sec', type=int, default=1) # number of points the neural net adjusts (evenly spaced in time)
parser.add_argument('--trajs', '-t', type=int, default=10)
parser.add_argument('--iterations', type=int, default=200)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--dt', type=float, default=0.002)
parser.add_argument('--input_weight', type=float, default=0) #weight on input in cost function
parser.add_argument('--loss_stride', type=float, default=10) # number of simulation steps before adding cost to loss again
parser.add_argument('--terminal_weight', type=float, default=100) 
parser.add_argument('--controller_stride', type=float, default=10)

parser.add_argument('--dubins_controller_weights', type=list, default=[3, 3, 3, 3])
parser.add_argument('--dubins_dyn_coeffs', type=list, default=[0.5, 0.25, 0.95, 0, 0]) #friction on v, phi, scale on inputs, init v between [0, v0] and phi between [-phi0, phi0]

parser.add_argument('--a1_controller_weights', type=list, default=[3, 3, 5, 5, 15]) #x, y, v, phi, w, alpha
parser.add_argument('--a1_warm_up_time', type=float, default=2)
parser.add_argument('--a1_warm_up_vel', type=list, default=[0.3, 0.5])

## A1 feasible traj have speed 0.1 to 1.5 and angle -0.15 to 0.15
## Car can do anything
parser.add_argument('--traj_v_range', type=list, default=[0.3, 0.6]) #velocity range for generated trajectories
parser.add_argument('--traj_theta_range', type=list, default=[-0.4, 0.4]) #theta range for generated trajectories
parser.add_argument('--traj_noise', type=float, default=0) #noise added to selected points (pulled from uniform [-noise, noise])

parser.add_argument('--model_scale', type=float, default=1)
parser.add_argument('--save_every', type=float, default=10) #save model every # iterations
parser.add_argument('--overwrite', '-o', action="store_true") #save model every # iterations

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


################### ENVIRONMENT SELECTION ######################
## Nominal Functions and Cost defined in helper.py

num_output_states = 2*params["points_per_sec"]*params["horizon"]

if params["env"] == "car":
    f_nominal = nominals["car"]
    weights = params["dubins_controller_weights"]
    coeffs = params["dubins_dyn_coeffs"]

    controller = Dubins_controller(k_x=weights[0], k_y=weights[1], k_v=weights[2], k_phi=weights[3])
    env = Dubins_env(total_time=params["horizon"], dt=params["dt"], f_v=coeffs[0], f_phi=coeffs[1], scale=coeffs[2],
                    v0=coeffs[3], phi0=coeffs[4])

    num_input_states = 2*params["horizon"] + 2 #v0 and phi0

elif params["env"] == "a1":
    f_nominal = nominals["a1"]
    weights = params["a1_controller_weights"]

    controller = A1_controller(k_x=weights[0], k_y=weights[1], k_v=weights[2], k_phi=weights[3], k_w=weights[4])
    env = A1GymEnv(total_time=params["horizon"], dt=params["dt"])
    #env = A1_env(total_time=params["horizon"], dt=params["dt"])

    num_input_states = 2*params["horizon"] + 3 #v0 and phi0 and phid0

else:
    raise NotImplementedError("Environment not implemented")

#################### NEURAL NETWORK SETUP ##########################
# Input: [x0, x1, ..., y0, y1, ..., v0, phi0]
# Output: Deltas on the above
# make_model in helper.py
model = make_model([num_input_states, 32, 32, num_output_states])

#Times the model affects
output_times = np.linspace(0, params["horizon"], params["points_per_sec"]*params["horizon"] + 1)

#################### TRAINING LOOP ##################################
optimizer = optim.Adam(model.parameters(), lr=params["lr"])

best_loss = np.inf
best_iter = 0
prog_bar = trange(params["iterations"], leave=True)
for i in prog_bar:

    optimizer.zero_grad() 

    # Collect trajectories (helper.py)
    dynamics, x0s, tasks, points_set = collect_trajs(model, env, controller, params, i)

    # Construct loss function
    loss = 0
    for (dyn, x0, task, points) in zip(dynamics, x0s, tasks, points_set):
        x = x0

        def f(x,u,t): 
            return f_nominal(x,u,params["dt"]) + dyn[t][2] - f_nominal(dyn[t][0],dyn[t][1],params["dt"]).detach()

        deltas = model(model_input(task, x0, params))*params["model_scale"]*(min(1, 2*(i+1)/params["iterations"]))
        task_adj = points + deltas
        spline = Spline(task_adj[:params["horizon"]*params["points_per_sec"]], task_adj[params["horizon"]*params["points_per_sec"]:], times=output_times, init_pos=x0)

        j = 0
        for t in np.arange(0, params["horizon"] + params["dt"], params["dt"]):
            if (j % params["controller_stride"] == 0): 
                u, des_pos, act_pos= controller.next_action(t, spline, x)

            if (j % params["loss_stride"] == 0):
                loss += cost(x, u, t, task, params, x0)
                
            x = f(x, u, int(t/params["dt"]))

            j += 1

    # Checkpoint
    loss_avg = loss.item() / (params["trajs"])
    if log: 
        writer.add_scalar("Loss/Train", loss_avg, i)

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
    loss.backward(retain_graph=True)
    optimizer.step()

    prog_bar.set_description("Loss: {}".format(loss_avg), refresh=True)


########################### FINAL LOGGING STUFF ###########################
if log:
    writer.close()
    if loss_avg < best_loss:
        best_iter = i
        torch.save(model, os.path.join(logdir, "best_model.pt"))
    torch.save(model, os.path.join(logdir, "final_model.pt"))
    with open(os.path.join(logdir, "params.json"), "w+") as outfile:
        json.dump(params, outfile)
    # with open(os.path.join(logdir, "best_model_iter.txt"), "w+") as outfile:
    #     outfile.write(str(best_iter))
    print("Best Iteration: ", best_iter)



