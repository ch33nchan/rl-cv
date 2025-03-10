import numpy as np

import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tinygrad"))
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear, Conv2d
from tinygrad.nn.optim import Adam

from .tinygrad_utils import to_tensor, to_numpy, create_lightweight_conv, create_lightweight_linear

class BaseModel:

    
    def __init__(self, input_shape, config=None):
        self.input_shape = input_shape
        self.config = config or {}
        self.model = None
        self.optimizer = None
        
    def build(self):
        raise NotImplementedError
        
    def setup_optimizer(self, learning_rate=0.001):

        if self.model is None:
            raise ValueError("Model must be built before setting up optimizer")
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        
    def forward(self, x):

        if self.model is None:
            raise ValueError("Model must be built before forward pass")
        return self.model(x)
    
    def save(self, path):
        """Save model weights"""
        if self.model is None:
            raise ValueError("Model must be built before saving")
      
        pass
    
    def load(self, path):
        """Load model weights"""
        if self.model is None:
            raise ValueError("Model must be built before loading weights")
      
        pass


class RLCVModel(BaseModel):
    
    def __init__(self, input_shape, action_space, config=None):
        super().__init__(input_shape, config)
        self.action_space = action_space
        self.conv_layers = []
        self.fc_layers = []
        
    def build(self):

        c, h, w = self.input_shape
        

        filters = self.config.get('filters', [16, 32, 32])
        kernel_sizes = self.config.get('kernel_sizes', [3, 3, 3])
        strides = self.config.get('strides', [1, 2, 2])
        fc_units = self.config.get('fc_units', [256])
        

        in_channels = c
        for i, (out_channels, kernel_size, stride) in enumerate(zip(filters, kernel_sizes, strides)):
            conv = create_lightweight_conv(in_channels, out_channels, kernel_size, stride)
            self.conv_layers.append(conv)
            in_channels = out_channels
            
    
            h = (h - kernel_size + 2*(kernel_size//2)) // stride + 1
            w = (w - kernel_size + 2*(kernel_size//2)) // stride + 1
        
     
        feature_size = h * w * filters[-1]
        
        in_features = feature_size
        for out_features in fc_units:
            fc = create_lightweight_linear(in_features, out_features)
            self.fc_layers.append(fc)
            in_features = out_features

        self.output_layer = create_lightweight_linear(in_features, self.action_space)
        
    def forward(self, x):

        x = to_tensor(x)

        for conv in self.conv_layers:
            x = conv(x).relu()
        

        x = x.reshape(x.shape[0], -1)
        

        for fc in self.fc_layers:
            x = fc(x).relu()
        

        x = self.output_layer(x)
        
        return x


class LightweightCNN(BaseModel):

    def build(self):
     
        c, h, w = self.input_shape
        

        self.conv1 = Conv2d(c, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        

        feature_size = (h // 4) * (w // 4) * 32
        
       
        output_size = self.config.get('output_size', 10)
        self.fc = Linear(feature_size, output_size)
        
    def forward(self, x):

        if not isinstance(x, Tensor):
            x = Tensor(x)
            
     
        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        x = self.conv3(x).relu()
        
 
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x


class RLCVModel:
 
    
    def __init__(self, input_shape, action_space, config=None):

        self.input_shape = input_shape
        self.action_space = action_space
        self.config = config or {}
        self.conv_layers = []
        self.fc_layers = []
        
    def build(self):

        c, h, w = self.input_shape
        

        filters = self.config.get('filters', [16, 32, 32])
        kernel_sizes = self.config.get('kernel_sizes', [3, 3, 3])
        strides = self.config.get('strides', [1, 2, 2])
        fc_units = self.config.get('fc_units', [256])
        

        in_channels = c
        for i, (out_channels, kernel_size, stride) in enumerate(zip(filters, kernel_sizes, strides)):
            conv = Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
            self.conv_layers.append(conv)
            in_channels = out_channels
            
 
            h = (h - kernel_size + 2*(kernel_size//2)) // stride + 1
            w = (w - kernel_size + 2*(kernel_size//2)) // stride + 1
        

        feature_size = h * w * filters[-1]
        
     
        in_features = feature_size
        for out_features in fc_units:
            fc = Linear(in_features, out_features)
            self.fc_layers.append(fc)
            in_features = out_features
            
    
        self.output_layer = Linear(in_features, self.action_space)
        
    def forward(self, x):

        if not isinstance(x, Tensor):
            x = Tensor(x)
        

        for conv in self.conv_layers:
            x = conv(x).relu()
        
        
        x = x.reshape(x.shape[0], -1)
        
       
        for fc in self.fc_layers:
            x = fc(x).relu()
        
       
        x = self.output_layer(x)
        
        return x