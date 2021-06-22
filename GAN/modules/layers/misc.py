# coding=utf-8
import torch
from torch.nn.modules.module import Module
import torch.nn as nn


class Conv2d_contiguous(nn.Conv2d):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
				 padding=0, dilation=1, groups=1, bias=True):
		super(Conv2d_contiguous, self).__init__(
			in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

	def forward(self, input):
		out = super(Conv2d_contiguous, self).forward(input)
		return out.contiguous()


class PrintSize(Module):
	def __init__(self):
		super(PrintSize, self).__init__()

	def forward(self, input):
		print(input.size())
		return input


class Dummy(Module):
	def __init__(self):
		super(Dummy, self).__init__()

	def forward(self, input):
		return input
