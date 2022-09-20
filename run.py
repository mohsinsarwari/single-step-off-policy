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
#from a1_learning_hierarchical.motion_imitation.envs.a1_env import A1GymEnv

log = True

LOG_PATH = "./logs"
BOARD_LOG_PATH = os.path.join(LOG_PATH, "tensorboard_logs")

RUN_NAME = "full_car_test2"

logdir = os.path.join(LOG_PATH, RUN_NAME)

#tensorboard
if log:
    writer = SummaryWriter(log_dir=os.path.join(BOARD_LOG_PATH, RUN_NAME))

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default="car")
parser.add_argument('--horizon', type=int, default=3)
parser.add_argument('--trajs', '-t', type=int, default=30)
parser.add_argument('--iterations', type=int, default=300)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--dt', type=float, default=0.01)
parser.add_argument('--input_weight', type=float, default=0)
parser.add_argument('--initial_pos', type=list, default=[0, 0])

args  = parser.parse_args()

params = vars(args)

params["model_scale"] = 3

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("-------Using {} ----------".format(dev))

# Setup problem parameters
f_nominal = None
cost = None
controller = None
if params["env"] == "car":
    def f_nominal(x, u): 
        x_clone = x.clone()
        x_clone[0] = x[0] + x[2] * torch.cos(x[3]) * params["dt"]
        x_clone[1] = x[1] + x[2] * torch.sin(x[3]) * params["dt"]
        x_clone[2] = x[2] + u[0] * params["dt"]
        x_clone[3] = x[3] + u[1] * params["dt"]

        return x_clone

    def cost(x, u, t, task):

        spline = Spline(task[:params["horizon"]], task[params["horizon"]:-2], xd_f=task[-2], yd_f=task[-1], dev=dev)
        x_d, y_d = spline.evaluate(t, der=0)

        return ((x[0] - x_d)**2 + (x[1] - y_d)**2) + (params["input_weight"] * (u[0]**2 + u[1]**2))

    controller = Dubins_controller(k_x=3, k_y=3, k_v=3, k_phi=3)
    env = Dubins_env(total_time=params["horizon"], dt=params["dt"], dev=dev)

elif params["env"] == "a1":
    def f_nominal(x, u): 
        x_clone = x.clone()
        x_clone[0] = x[0] + x[2] * torch.cos(x[3]) * params["dt"]
        x_clone[1] = x[1] + x[2] * torch.sin(x[3]) * params["dt"]
        x_clone[2] = x[2] + u[0] * params["dt"]
        x_clone[3] = x[3] + x[4] * params["dt"]
        x_clone[4] = x[4] + u[1] * params["dt"]

        return x_clone

    def cost(x, u, t, task):

        spline = Spline(task[:params["horizon"]], task[params["horizon"]:-2], xd_f=task[-2], yd_f=task[-1], dev=dev)
        x_d, y_d = spline.evaluate(t, der=0)

        return ((x[0] - x_d)**2 + (x[1] - y_d)**2) + (params["input_weight"] * (u[0]**2 + u[1]**2))

    #controller = A1_controller(5, 5, 35, 5, 30)
    #env = A1GymEnv(total_time=params["horizon"])

else:
    raise NotImplementedError("Environment not implemented")

# Setup NN
# Input: [x0, x1, ..., y0, y1, ..., xd_f, yd_f]
# Output: Deltas on the above
model = make_model([2*params["horizon"]+2, 32, 32, 2*params["horizon"]+2])
model.to(dev)

# Collect trajectories function
def collect_trajs(model):

    with torch.no_grad():
        trajs = []
        x0s = []
        tasks = []
        for i in range(params["trajs"]):
            task = generate_traj(params["horizon"], dev=dev)
            tasks.append(task)
            deltas = model(task)*params["model_scale"]
            task_adj = task + deltas
            spline = Spline(task_adj[:params["horizon"]], task_adj[params["horizon"]:-2], xd_f=task_adj[-2], yd_f=task_adj[-1], dev=dev)
            traj = []
            obs = env.reset()
            x0s.append(obs)
            for j in np.arange(0, params["horizon"], params["dt"]):
                action, _, _ = controller.next_action(j, spline, obs)
                next_obs, reward, done, info = env.step(action)
                traj.append((obs, action, reward, next_obs, done))
                obs = next_obs
            trajs.append(traj)

        # Make time-varying dynamics
        fs = []
        for traj in trajs:
            dyn_tuples = []
            for obs, action, reward, next_obs, done in traj:
                dyn_tuples.append((obs, action, next_obs))
            fs.append(dyn_tuples)

    return fs, x0s, tasks

# Training Loop
optimizer = optim.Adam(model.parameters(), lr=params["lr"])
prog_bar = trange(params["iterations"], leave=True)
for i in prog_bar:#tqdm(range(params["iterations"])):
    optimizer.zero_grad() 

    # Collect trajectories
    dynamics, x0s, tasks = collect_trajs(model)

    # Construct loss function
    loss = 0
    for (dyn, x0, task) in zip(dynamics, x0s, tasks):

        x = x0

        def f(x,u,t): 
            return f_nominal(x,u) + dyn[t][2] - f_nominal(dyn[t][0], dyn[t][1]).detach()

        deltas = model(task)*params["model_scale"]
        task_adj = task + deltas
        spline = Spline(task_adj[:params["horizon"]], task_adj[params["horizon"]:-2], xd_f=task_adj[-2], yd_f=task_adj[-1], dev=dev)
        for t in np.arange(0, params["horizon"], params["dt"]):
            u, des_pos, act_pos= controller.next_action(t, spline, x)

            if t % (3*params["dt"]) == 0:
                loss += cost(x, u, t, task)

            x = f(x, u, int(t/params["dt"]))

    prog_bar.set_description("Loss: {}".format(loss), refresh=True)

    # Backprop
    if log:
        writer.add_scalar("Loss/Train", loss.item(), i)
    loss.retain_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

if log:
    os.mkdir(logdir)
    writer.close()
    torch.save(model.cpu(), os.path.join(logdir, "model.pt"))

    with open(os.path.join(logdir, "params.json"), "w+") as outfile:
        json.dump(params, outfile)



