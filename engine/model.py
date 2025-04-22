import numpy as np

import sys
import os

# Adjust path if tinygrad is not installed globally or in the parent directory
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tinygrad"))

try:
    from tinygrad.tensor import Tensor
    from tinygrad.nn.optim import Adam
    from tinygrad.nn.state import load_state_dict, get_parameters
except ImportError:
    print("Error: Tinygrad not found. Make sure it's installed or the path is correct.")
    sys.exit(1)

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


class ReasoningModel:
    """
    Base class for Visual/Video Language Models focused on reasoning,
    built on TinyGrad.
    """
    def __init__(self, config=None):
        self.config = config or {}
        self.model_components = {} # Dictionary to hold parts like vision encoder, language model, etc.
        self.optimizer = None
        # TODO: Define expected config parameters (e.g., base model name, embed_dim)

    def build(self):
        """
        Construct the model components based on the config.
        This might involve loading parts of a pre-trained model.
        """
        raise NotImplementedError("Subclasses must implement the build method")

    def setup_optimizer(self, learning_rate=1e-4):
        """Sets up the Adam optimizer for trainable parameters."""
        if not self.model_components:
            raise ValueError("Model must be built before setting up optimizer")

        trainable_params = []
        for component in self.model_components.values():
             # Assuming components might be simple functions or non-parameterized objects
             if hasattr(component, 'parameters'):
                 trainable_params.extend(get_parameters(component))
             elif isinstance(component, Tensor) and component.requires_grad:
                 trainable_params.append(component)

        # Filter out duplicates if parameters are shared across components
        unique_params = list({id(p): p for p in trainable_params}.values())

        if not unique_params:
             print("Warning: No trainable parameters found for the optimizer.")
             self.optimizer = None
        else:
             self.optimizer = Adam(unique_params, lr=learning_rate)
        print(f"Optimizer set up for {len(unique_params)} parameters.")


    def forward(self, *args, **kwargs):
        """
        Defines the forward pass of the model.
        Inputs might include image tensors, text tokens, etc.
        """
        raise NotImplementedError("Subclasses must implement the forward method")

    def get_trainable_parameters(self):
        """Helper to get all trainable parameters from components."""
        params = []
        for component in self.model_components.values():
            if hasattr(component, 'parameters'): # Check if component has tinygrad parameters
                 params.extend(get_parameters(component))
            elif isinstance(component, Tensor) and component.requires_grad:
                 params.append(component) # Handle standalone trainable tensors
        # Return unique parameters
        return list({id(p): p for p in params}.values())


    def load_base_weights(self, path):
        """Loads weights from a pre-trained base model file."""
        # This will need careful implementation based on how base models are stored
        # and how they map to your model_components structure.
        print(f"Loading base weights from {path} (Implementation needed)")
        # Example: load_state_dict(self.model_components['language_model'], safe_load(path))
        pass

    def save_weights(self, path):
        """Saves the current model weights."""
        # Implementation needed using tinygrad.nn.state.safe_save
        pass

    # Add other necessary methods like generate (for text output), train_step, etc.

# Example of a specific implementation (highly simplified)
class SimpleVLM(ReasoningModel):
    def build(self):
        # Placeholder: In reality, load/define actual TinyGrad layers
        print("Building SimpleVLM...")
        # These would be actual tinygrad modules (Linear, Conv2d, etc.) or loaded models
        self.model_components['vision_encoder'] = lambda x: x # Dummy vision encoder
        self.model_components['language_model'] = lambda x: x # Dummy language model
        print("SimpleVLM built.")

    def forward(self, image_input: Tensor, text_input: Tensor):
        print("SimpleVLM forward pass...")
        vision_features = self.model_components['vision_encoder'](image_input)
        # Combine features and process through language model (highly simplified)
        output = self.model_components['language_model'](text_input + vision_features.mean()) # Dummy combination
        return output