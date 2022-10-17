import torch
import numpy as np
import matplotlib.pyplot as plt
import pdb

from spline import Spline




def figure_eight(xd_0=0, yd_0=0, xd_f=None, yd_f=None, radius=1, time=None):

	x_points = [radius, 2*radius, radius, 0, -radius, -2*radius, -radius, 0]
	y_points = [radius, 0, -radius, 0, radius, 0, -radius, 0]

	if time is None:
		times = None
	else:
		times = np.linspace(0, time, 9)

	return Spline(x_points, y_points, xd_0=0, yd_0=0, xd_f=None, yd_f=None, times=times, init_pos=torch.tensor([0, 0], dtype=torch.float))


def random(time=None):

	x_points = [1, 2, 1]
	y_points = [1, 2, 3]

	if time is None:
		times = None
	else:
		times = np.linspace(0, time, 4)

	return Spline(x_points, y_points, xd_0=0, yd_0=0, xd_f=None, yd_f=None, times=times, init_pos=torch.tensor([0, 0], dtype=torch.float))

if __name__=="__main__":
	time = 9
	#cs = figure_eight(xd_0=0.5, yd_0=0.5, xd_f=0.5, yd_f=0.5, radius=10, time=time)
	cs = random(time=time)
	times = np.linspace(0, time, 80)

	xs = []
	ys = []

	for time in times:
		x, y = cs.evaluate(time)
		xs.append(x)
		ys.append(y)

	plt.plot(xs, ys)
	plt.show()

