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
		linear = nn.Linear(layer_sizes[i], layer_sizes[i+1])

		if i == len(layer_sizes) - 2:
			linear.weight.data.fill_(0)
			linear.bias.data.fill_(0)

		layers.append(linear)
		layers.append(nn.Tanh())
	return nn.Sequential(*layers)

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

def cost(obs, u, t, task, params):

	x_d, y_d = task.evaluate(t, der=0)

	ret = ((obs[0] - x_d)**2 + (obs[1] - y_d)**2) + (params["input_weight"] * (u[0]**2 + u[1]**2))
		
	return ret

def model_input(obs, t, task_time):
	res = torch.hstack((obs, torch.tensor([np.cos(2*np.pi*t/task_time), np.sin(2*np.pi*t/task_time)], dtype=torch.float)))
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



