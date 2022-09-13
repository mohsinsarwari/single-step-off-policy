import torch
import torch.nn as nn
import numpy as np

def make_model(layer_sizes):
	layers = []
	for i in range(len(layer_sizes) - 1):
		layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
		layers.append(nn.Tanh())
	return nn.Sequential(*layers)
	