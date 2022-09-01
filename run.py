import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym

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
traj_length = args.trajs
env_name = args.env
seed = args.seed
epochs = args.epochs
lr = args.lr
loops = args.loops

# Setup problem parameters
f_nominal = None
cost = None
make_controller = None
if env_name == "car": # TODO: Check if this is correct
    f_nominal = lambda x, u: np.array([x[0] + x[2] * np.cos(x[3]) * u[0],
                                       x[1] + x[2] * np.sin(x[3]) * u[0],
                                       x[2] + u[1],
                                       x[3] + u[0]])
    cost = lambda x, u, t: 0.1 * u[0] ** 2 + 0.1 * u[1] ** 2 + 0.1 * (x[2] - 1) ** 2

    def make_controller(spline):
        def controller(x):
            return 0 # TODO: implement
        return controller
else:
    raise NotImplementedError("Environment not implemented")

# Collect trajectories function
def collect_trajs(spline):
    env = gym.make(env_name)
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    controller = make_controller(spline)

    trajs = []
    x0s = []
    for i in range(traj_length):
        traj = []
        obs = env.reset()
        x0s.append(obs)
        for j in range(horizon):
            action = controller(obs)
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
        def f(x, u, t):
            return dyn_tuples[t][2] + f_nominal(dyn_tuples[t][0], dyn_tuples[t][1]) - x - u # TODO: fix
        fs.append(f)
    return fs, x0s

# Setup NN
model = nn.Linear(1, 1) # TODO: Setup right model

# Train NN
for loop in range(loops):
    # Collect trajectories
    dynamics, x0s = collect_trajs(model)
    controller = make_controller(model)

    # Construct loss function
    loss_fn = 0
    for (f, x0) in zip(dynamics, x0s):
        x = x0
        for t in range(horizon):
            u = controller(x)
            loss_fn += cost(x, u, t)
            x = f(x, u, t)

    # Backprop
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Run training
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(x)
        loss = loss_fn(predictions, t)
        loss.backward()
        optimizer.step()
