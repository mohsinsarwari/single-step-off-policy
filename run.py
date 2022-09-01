import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from helper import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default="car")
parser.add_argument('--horizon', '-h', type=int, default=100)
parser.add_argument('--trajs', '-t', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--loops', type=int, default=10)
parser.add_argument('--epochs', '-e', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.05)

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

# Setup problem parameters
f_nominal = None
cost = None
controller = None
if env_name == "car": # TODO: Check if this is correct
    f_nominal = lambda x, u: np.array([x[0] + x[2] * np.cos(x[3]) * u[0],
                                       x[1] + x[2] * np.sin(x[3]) * u[0],
                                       x[2] + u[1],
                                       x[3] + u[0]])
    cost = lambda x, u, t, spline: 0.1 * u[0] ** 2 + 0.1 * u[1] ** 2 + 0.1 * (x[2] - 1) ** 2

    def controller(x, t, spline):
        return 0 # TODO: implement

    def sample_task():
        return np.random.uniform(0.5, 1.5, 1) # TODO: Fix this
else:
    raise NotImplementedError("Environment not implemented")

# Collect trajectories function
def collect_trajs(model):
    env = gym.make(env_name)
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    trajs = []
    x0s = []
    tasks = []
    for i in range(num_trajs):
        task = sample_task()
        tasks.append(task)
        spline = model(task)
        traj = []
        obs = env.reset()
        x0s.append(obs)
        for j in range(horizon):
            action = controller(obs, j, spline)
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
model = make_model([1, 32, 32, 4]) # TODO: Setup write layer sizes

# Training Loop
for loop in range(loops):
    # Collect trajectories
    dynamics, x0s, tasks = collect_trajs(model)

    # Construct loss function
    loss = 0
    for (dyn, x0, task) in zip(dynamics, x0s, tasks):
        x = x0
        f = lambda x,u,t: f_nominal(x,u) + dyn[t][2] - f_nominal(dyn[t][0], dyn[t][1])
        spline = model(task)
        for t in range(horizon):
            u = controller(x, t, spline)
            loss += cost(x, u, t, task)
            x = f(x, u, t)

    # Backprop
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Test NN
