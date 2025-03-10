import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tinygrad"))
from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d, Linear

def to_tensor(x):
    if isinstance(x, Tensor):
        return x
    return Tensor(x)

def to_numpy(x):

    if isinstance(x, Tensor):
        return x.numpy()
    return x

def create_lightweight_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):

    return Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

def create_lightweight_linear(in_features, out_features):

    return Linear(in_features, out_features)