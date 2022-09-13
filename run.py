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

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default="car")
parser.add_argument('--horizon', type=int, default=4)
parser.add_argument('--trajs', '-t', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--loops', type=int, default=1)
parser.add_argument('--epochs', '-e', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--total_time', type=float, default=4)
parser.add_argument('--dt', type=float, default=0.01)
parser.add_argument('--input_weight', type=float, default=0.5)

## Example arg parsers arguments
# parser.add_argument('--epochs_reward', type=int, default=3)
# parser.add_argument('--epochs_agent', type=int, default=10)
# parser.add_argument('--comparisons', type=int, default=5000)
# parser.add_argument('--algo', type=str, default="ppo")
# parser.add_argument('--seed', type=int, default=0)
# parser.add_argument('--fragment_length', type=int, default=50)
# parser.add_argument('--pref', action='store_true')
# parser.add_argument('--render', action='store_true')

args  = parser.parse_args()
horizon = args.horizon
num_trajs = args.trajs
env_name = args.env
seed = args.seed
epochs = args.epochs
lr = args.lr
loops = args.loops
total_time = args.total_time
dt = args.dt
times = np.arange(total_time+1)
input_weight = args.input_weight

# Setup problem parameters
f_nominal = None
cost = None
controller = None
if env_name == "car":
    f_nominal = lambda x, u: torch.tensor([x[0] + x[2] * torch.cos(x[3]) * dt,
                                       x[1] + x[2] * torch.sin(x[3]) * dt,
                                       x[2] + u[0] * dt,
                                       x[3] + u[1] * dt], dtype=torch.float)

    def cost(x, u, t, spline_params):

        spline = Spline(times, spline_params[:total_time+1], spline_params[total_time+1:])
        x_d, y_d = spline.evaluate(t, der=0)

        return ((x[0] - x_d)**2 + (x[1] - y_d)**2) + (input_weight * (u[0]**2 + u[1]**2))

    def sample_task():
        return [0, 1, 2, 1, 0, 0, 1, 0, -1, 0] #torch.tensor([0, 1, 2, 1, 0, 0, 1, 0, -1, 0], dtype=torch.float)

    controller = Dubins_controller(k_x=12, k_y=12, k_v=12, k_phi=12)
    env = Dubins_env(total_time=total_time, dt=dt)

else:
    raise NotImplementedError("Environment not implemented")

# Collect trajectories function
def collect_trajs(model):
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    trajs = []
    x0s = []
    tasks = []
    for i in range(num_trajs):
        task = sample_task()
        tasks.append(task)
        spline_params = model(torch.tensor(task, dtype=torch.float, requires_grad=False))
        spline = Spline(times, spline_params[:total_time+1], spline_params[total_time+1:])
        traj = []
        obs = env.reset()
        x0s.append(obs)
        for j in np.arange(0, total_time, dt):
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

# Setup NN
torch.autograd.set_detect_anomaly(True)
model = make_model([2*len(times), 32, 32, 2*len(times)])

# Training Loop
for loop in range(loops):
    # Collect trajectories
    dynamics, x0s, tasks = collect_trajs(model)

    # Construct loss function
    loss = 0
    for (dyn, x0, task) in zip(dynamics, x0s, tasks):
        x = x0
        f = lambda x,u,t: f_nominal(x,u) + dyn[t][2] - f_nominal(dyn[t][0], dyn[t][1])
        spline_params = model(torch.tensor(task, dtype=torch.float))
        spline = Spline(times, spline_params[:total_time+1], spline_params[total_time+1:])
        for t in np.arange(0, total_time, dt):
            u, _, _ = controller.next_action(t, spline, x)
            loss += cost(x, u, t, task)
            x = f(x.clone(), u, int(t//dt))

    # Backprop
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

# Test NN
