import torch
import torch.nn as nn
import numpy as np
from spline import Spline
import matplotlib
import matplotlib.pyplot as plt
import pdb

def make_model(layer_sizes):
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)

def sample_task():
    task = torch.tensor([1, 2, 3, 1, 2, 3, 0, 0], dtype=torch.float)
    return task

def generate_traj(horizon=5, noise=0.6, v_range=[1, 4], theta_range=[-np.pi/4, np.pi/4]):
    xs = []
    ys = []
    dt = 0.1

    v = np.random.uniform(v_range[0],v_range[1])
    theta = np.random.uniform(theta_range[0],theta_range[1])

    x = 0
    y = 0
    psi = 0

    for i in range(int(horizon/dt) + 1):

        x_dot = v * np.cos(psi)
        y_dot = v * np.sin(psi)
        psi_dot = theta

        x += dt*x_dot
        y += dt*y_dot
        psi += dt*psi_dot

        if (i % int(1/dt) == 0) and (i != 0):
            x += np.random.uniform(-noise, noise)
            y += np.random.uniform(-noise, noise)
            xs.append(x)
            ys.append(y)

    res = torch.hstack((torch.tensor(xs, dtype=torch.float), torch.tensor(ys, dtype=torch.float)))

    return res

def find_points(task, params):
    spline = Spline(task[:params["horizon"]], task[params["horizon"]:])
    ts = np.linspace(0, params["horizon"], params["points_per_sec"]*params["horizon"] + 1)
    xs = []
    ys = []

    for t in ts[1:]: #ignore 0
        x, y = spline.evaluate(t)
        xs.append(x)
        ys.append(y)

    return torch.hstack((torch.tensor(xs, dtype=torch.float), torch.tensor(ys, dtype=torch.float)))

def car_nominal(x, u, dt): 
    x_clone = x.clone()
    x_clone[0] = x[0] + x[2] * torch.cos(x[3]) * dt
    x_clone[1] = x[1] + x[2] * torch.sin(x[3]) * dt
    x_clone[2] = x[2] + u[0] * dt
    x_clone[3] = x[3] + u[1] * dt

    return x_clone

def a1_nominal(x, u, dt): 
    x_clone = x.clone()
    x_clone[0] = x[0] + x[2] * torch.cos(x[3]) * dt
    x_clone[1] = x[1] + x[2] * torch.sin(x[3]) * dt
    x_clone[2] = x[2] + u[0] * dt
    x_clone[3] = x[3] + x[4] * dt
    x_clone[4] = x[4] + u[1] * dt

    return x_clone

nominals = {"car": car_nominal, "a1": a1_nominal}

def cost(x, u, t, task, params, init_pos):

    spline = Spline(task[:params["horizon"]], task[params["horizon"]:], init_pos=init_pos)
    x_d, y_d = spline.evaluate(t, der=0)

    return ((x[0] - x_d)**2 + (x[1] - y_d)**2) + (params["input_weight"] * (u[0]**2 + u[1]**2))

def model_input(task, obs, params):
    if params["env"] == "car":
        res = torch.hstack((task, obs[2:]))
    if params["env"] == "a1":
        res = torch.hstack((task, obs[2:]))
    return res

def a1_warm_up(env, controller, params):
    obs = env.reset()
    env.warm_up = True
    v_des = torch.tensor(np.random.uniform(params["a1_warm_up_vel"][0], params["a1_warm_up_vel"][1]), dtype=torch.float)
    phi_des = torch.tensor(0, dtype=torch.float)
    for _ in np.arange(0, params["a1_warm_up_time"], params["dt"]):
        action = controller.next_action_warm_up(v_des, phi_des, obs)
        obs, reward, done, info = env.step(action)
    env.init_time = env.robot.GetTimeSinceReset()
    env.warm_up = False
    print("--------------WARMED UP----------------")
    return obs

def heading_arrays(xs, ys, phis, stride=5):
    return xs[::stride], ys[::stride], [np.cos(phi) for phi in phis[::stride]], [np.sin(phi) for phi in phis[::stride]] 


def collect_trajs(model, env, controller, params, i):

    output_times = np.linspace(0, params["horizon"], params["points_per_sec"]*params["horizon"] + 1)

    with torch.no_grad():
        trajs = []
        x0s = []
        tasks = []
        points_set = []
        for i in range(params["trajs"]):
            task = generate_traj(params["horizon"], params["traj_noise"], params["traj_v_range"], params["traj_theta_range"])
            tasks.append(task)
            if params["env"] == "car":
                obs = env.reset()
            elif params["env"] == "a1":
                obs = a1_warm_up(env, controller, params)
            x0s.append(obs)
            points = find_points(task, params)
            points_set.append(points)
            deltas = model(model_input(task, obs, params))*params["model_scale"]*(min(1, 2*(i+1)/params["iterations"]))
            task_adj = points + deltas
            spline = Spline(task_adj[:params["horizon"]*params["points_per_sec"]], task_adj[params["horizon"]*params["points_per_sec"]:], times=output_times, init_pos=obs)
            traj = []
            k = 0
            for j in np.arange(0, params["horizon"] + params["dt"], params["dt"]):
                if (k % params["controller_stride"] == 0):
                    action, des_pos, act_pos = controller.next_action(j, spline, obs)
                next_obs, reward, done, info = env.step(action)
                traj.append((obs, action, reward, next_obs, done))
                obs = next_obs
                k += 1
            trajs.append(traj)

        # Make time-varying dynamics
        fs = []
        for traj in trajs:
            dyn_tuples = []
            for obs, action, reward, next_obs, done in traj:
                dyn_tuples.append((obs, action, next_obs))
            fs.append(dyn_tuples)

    return fs, x0s, tasks, points_set


if __name__ == "__main__":

    horizon = 3

    for i in range(10):

        task = generate_traj(horizon, 0, [1, 5], [-1, 1])

        find_points(task, {"horizon": horizon, "dt": 0.01, "points_per_sec": 2})

        cs = Spline(task[:horizon], task[horizon:])

        times = np.linspace(0, horizon, 40)
        xs = []
        ys = []
        for time in times:
            x, y = cs.evaluate(time)
            xs.append(x)
            ys.append(y)

        plt.plot(xs, ys)

    plt.show()



