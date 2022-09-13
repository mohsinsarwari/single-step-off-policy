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
parser.add_argument('--loops', type=int, default=100)
parser.add_argument('--epochs', '-e', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--total_time', type=float, default=4)
parser.add_argument('--dt', type=float, default=0.01)
parser.add_argument('--input_weight', type=float, default=0)

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
model_scale = 5

# Setup problem parameters
f_nominal = None
cost = None
controller = None
if env_name == "car":
    def f_nominal(x, u): 
        x_clone = x.clone()
        x_clone[0] = x[0] + x[2] * torch.cos(x[3]) * dt
        x_clone[1] = x[1] + x[2] * torch.sin(x[3]) * dt
        x_clone[2] = x[2] + u[0] * dt
        x_clone[3] = x[3] + u[1] * dt

        return x_clone

    def cost(x, u, t, spline_params):

        spline = Spline(times, spline_params[1:total_time+1], spline_params[total_time+2:], spline_params[0], spline_params[total_time+1])
        x_d, y_d = spline.evaluate(t, der=0)

        return ((x[0] - x_d)**2 + (x[1] - y_d)**2) + (input_weight * (u[0]**2 + u[1]**2))

    def sample_task():
        #task = torch.tensor([0, 1, 2, 0, 1, 0], dtype=torch.float) #torch.tensor([0, 1, 2, 1, 0, 0, 1, 0, -1, 0], dtype=torch.float)
        task = torch.tensor([0, 1, 2, 1, 0, 0, 1, 0, -1, 0], dtype=torch.float)
        return task

    controller = Dubins_controller(k_x=5, k_y=5, k_v=5, k_phi=5)
    env = Dubins_env(total_time=total_time, dt=dt)

else:
    raise NotImplementedError("Environment not implemented")

# Collect trajectories function
def collect_trajs(model):

    with torch.no_grad():
        # env.seed(seed)
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        trajs = []
        x0s = []
        tasks = []
        for i in range(num_trajs):
            task = sample_task()
            tasks.append(task)
            spline_params = model(task)*model_scale
            spline = Spline(times, spline_params[:total_time], spline_params[total_time:])
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
model = make_model([2*len(times), 32, 32, 2*(len(times)-1)])

# Training Loop

optimizer = optim.Adam(model.parameters(), lr=lr)
for loop in range(loops):
    optimizer.zero_grad() 

    # Collect trajectories
    dynamics, x0s, tasks = collect_trajs(model)

    # Construct loss function
    loss = 0
    for (dyn, x0, task) in zip(dynamics, x0s, tasks):
        x = x0
        def f(x,u,t): 
            return f_nominal(x,u) + dyn[t][2] - f_nominal(dyn[t][0], dyn[t][1]).detach()
        spline_params = model(task)*model_scale

        des_x = []
        des_y = []
        act_x = []
        act_y = []
        tar_x = []
        tar_y = []

        spline = Spline(times, spline_params[:total_time], spline_params[total_time:])
        for t in np.arange(0, total_time, dt):
            u, des_pos, act_pos= controller.next_action(t, spline, x)

            target = Spline(times, task[1:total_time+1], task[total_time+2:], task[0], task[total_time+1])
            tar_pos_x, tar_pos_y = target.evaluate(t, der=0)

            loss += cost(x, u, t, task)
            x = f(x, u, int(t//dt))

            des_x.append(des_pos[0].detach().numpy())
            des_y.append(des_pos[1].detach().numpy())
            act_x.append(act_pos[0].detach().numpy())
            act_y.append(act_pos[1].detach().numpy())
            tar_x.append(tar_pos_x.detach().numpy())
            tar_y.append(tar_pos_y.detach().numpy())

    # Backprop
    # for epoch in range(epochs): 
    loss.retain_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    # with torch.no_grad():
    #     for p in model.parameters():
    #         new_val = p - p.grad*lr
    #         p.copy_(new_val)
    print("Loss: ", loss)


with torch.no_grad():
    plt.plot(des_x, des_y, label="model output")
    plt.plot(act_x, act_y, label="actual")
    plt.plot(tar_x, tar_y, label="desired")
    plt.legend()
    plt.show()
    #print("Params Diff: ", np.linalg.norm(np.array([x.detach().numpy() for x in model.parameters()]) - last))

# Test NN
