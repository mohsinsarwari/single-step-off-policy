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

		x_prev = x
		y_prev = y

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

	xd_f = (x - x_prev)/dt
	yd_f = (y - y_prev)/dt

	res = torch.hstack((torch.tensor(xs, dtype=torch.float), torch.tensor(ys, dtype=torch.float), 
				torch.tensor(xd_f, dtype=torch.float), torch.tensor(yd_f, dtype=torch.float)))

	return res

def find_points(task, params):
	spline = Spline(task[:params["horizon"]], task[params["horizon"]:-2], xd_f=task[-2], yd_f=task[-1])
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

def cost(x, u, t, task, params):

    spline = Spline(task[:params["horizon"]], task[params["horizon"]:-2], xd_f=task[-2], yd_f=task[-1])
    x_d, y_d = spline.evaluate(t, der=0)

    return ((x[0] - x_d)**2 + (x[1] - y_d)**2) + (params["input_weight"] * (u[0]**2 + u[1]**2))


def collect_trajs(model, env, controller, params):

    output_times = np.linspace(0, params["horizon"], params["points_per_sec"]*params["horizon"] + 1)

    with torch.no_grad():
        trajs = []
        x0s = []
        tasks = []
        points_set = []
        for i in range(params["trajs"]):
            task = generate_traj(params["horizon"], params["traj_noise"], params["traj_v_range"], params["traj_theta_range"])
            tasks.append(task)
            points = find_points(task, params)
            points_set.append(points)
            deltas = model(task[:-2])*params["model_scale"]
            task_adj = points + deltas
            spline = Spline(task_adj[:params["horizon"]*params["points_per_sec"]], task_adj[params["horizon"]*params["points_per_sec"]:], times=output_times)
            traj = []
            obs = env.reset()
            x0s.append(obs)
            for j in np.arange(0, params["horizon"], params["dt"]):
                action, des_pos, act_pos = controller.next_action(j, spline, obs)
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

    return fs, x0s, tasks, points_set


if __name__ == "__main__":

	horizon = 3

	for i in range(10):

		task = generate_traj(horizon, 0, [1, 5], [-1, 1], 2)

		find_points(task, {"horizon": horizon, "dt": 0.01, "points_per_sec": 2})

		cs = Spline(task[:horizon], task[horizon:-2], xd_f=task[-2], yd_f=task[-1])

		times = np.linspace(0, horizon, 40)
		xs = []
		ys = []
		for time in times:
			x, y = cs.evaluate(time)
			xs.append(x)
			ys.append(y)

		plt.plot(xs, ys)

	plt.show()



