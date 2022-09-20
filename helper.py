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

def generate_traj(horizon=5, noise=0.5):
	xs = []
	ys = []
	dt = 0.1

	v = np.random.uniform(1,5)
	theta = np.random.uniform(-np.pi/2,np.pi/2)

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

	return torch.hstack((torch.tensor(xs, dtype=torch.float), torch.tensor(ys, dtype=torch.float))), xd_f, yd_f


if __name__ == "__main__":

	horizon = 5

	for i in range(1):

		task, xd_f, yd_f = generate_traj(horizon)

		cs = Spline(task[:horizon], task[horizon:], xd_0=5, yd_0=2, xd_f=xd_f, yd_f=yd_f)

		times = np.linspace(0, horizon, 40)
		xs = []
		ys = []
		for time in times:
			x, y = cs.evaluate(time)
			xs.append(x)
			ys.append(y)

		plt.plot(xs, ys)

	plt.show()




