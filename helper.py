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
	task = torch.tensor([1, 2, 3, 1, 2, 3], dtype=torch.float)
	return task, 0, 0

def generate_traj(horizon=5, noise=0.6, dev=torch.device("cpu")):
	xs = []
	ys = []
	dt = 0.1

	v = np.random.uniform(1,4)
	theta = np.random.uniform(-np.pi/4,np.pi/4)

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

	return res.to(dev)


if __name__ == "__main__":

	horizon = 3

	for i in range(10):

		task = generate_traj(horizon)

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




