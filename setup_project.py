import os
import sys

def create_directory_structure():
    """Create the basic directory structure for the project"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define directories to create
    directories = [
        'core',
        'rl',
        'vision',
        'utils',
        'examples',
        'docs',
        'tests',
        'models'
    ]
    
    # Create directories
    for directory in directories:
        dir_path = os.path.join(base_dir, directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")
        else:
            print(f"Directory already exists: {dir_path}")
    
    # Create __init__.py files
    for directory in directories:
        init_file = os.path.join(base_dir, directory, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write("# Initialize package\n")
            print(f"Created __init__.py in {directory}")

def check_required_files():
    """Check if all required files exist"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    required_files = [
        ('core/model.py', create_model_file),
        ('core/tinygrad_utils.py', create_tinygrad_utils_file),
        ('rl/agent.py', create_agent_file),
        ('rl/environment.py', create_environment_file),
        ('rl/memory.py', create_memory_file),
        ('rl/policy.py', create_policy_file),
        ('vision/preprocessing.py', create_preprocessing_file)
    ]
    
    for file_path, create_func in required_files:
        full_path = os.path.join(base_dir, file_path)
        if not os.path.exists(full_path):
            print(f"Creating missing file: {full_path}")
            create_func(full_path)
        else:
            print(f"File exists: {full_path}")

# File creation functions
def create_model_file(path):
    with open(path, 'w') as f:
        f.write("""import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear, Conv2d

class BaseModel:
    \"\"\"Base model class for all RLCV models\"\"\"
    
    def __init__(self, input_shape, config=None):
        \"\"\"
        Initialize the base model
        
        Args:
            input_shape: Shape of input tensor (channels, height, width)
            config: Configuration dictionary
        \"\"\"
        self.input_shape = input_shape
        self.config = config or {}
        self.model = None
        self.optimizer = None
        
    def build(self):
        \"\"\"Build the model architecture - to be implemented by subclasses\"\"\"
        raise NotImplementedError
        
    def forward(self, x):
        \"\"\"Forward pass through the model\"\"\"
        raise NotImplementedError


class RLCVModel(BaseModel):
    \"\"\"Reinforcement Learning Computer Vision model using TinyGrad\"\"\"
    
    def __init__(self, input_shape, action_space, config=None):
        \"\"\"
        Initialize the RLCV model
        
        Args:
            input_shape: Shape of input tensor (channels, height, width)
            action_space: Number of possible actions
            config: Configuration dictionary
        \"\"\"
        super().__init__(input_shape, config)
        self.action_space = action_space
        self.conv_layers = []
        self.fc_layers = []
        
    def build(self):
        \"\"\"Build a CNN architecture for RL tasks\"\"\"
        c, h, w = self.input_shape
        
        # Configuration for the model
        filters = self.config.get('filters', [16, 32, 32])
        kernel_sizes = self.config.get('kernel_sizes', [3, 3, 3])
        strides = self.config.get('strides', [1, 2, 2])
        fc_units = self.config.get('fc_units', [256])
        
        # Build convolutional layers
        in_channels = c
        for i, (out_channels, kernel_size, stride) in enumerate(zip(filters, kernel_sizes, strides)):
            conv = Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
            self.conv_layers.append(conv)
            in_channels = out_channels
            
            # Calculate output dimensions
            h = (h - kernel_size + 2*(kernel_size//2)) // stride + 1
            w = (w - kernel_size + 2*(kernel_size//2)) // stride + 1
        
        # Calculate flattened feature size
        feature_size = h * w * filters[-1]
        
        # Build fully connected layers
        in_features = feature_size
        for out_features in fc_units:
            fc = Linear(in_features, out_features)
            self.fc_layers.append(fc)
            in_features = out_features
            
        # Output layer for action values
        self.output_layer = Linear(in_features, self.action_space)
        
    def forward(self, x):
        \"\"\"Forward pass through the model\"\"\"
        # Ensure input is a Tensor
        if not isinstance(x, Tensor):
            x = Tensor(x)
        
        # Apply convolutional layers with ReLU
        for conv in self.conv_layers:
            x = conv(x).relu()
        
        # Flatten
        x = x.reshape(x.shape[0], -1)
        
        # Apply fully connected layers with ReLU
        for fc in self.fc_layers:
            x = fc(x).relu()
        
        # Output layer (no activation for Q-values)
        x = self.output_layer(x)
        
        return x
""")

def create_tinygrad_utils_file(path):
    with open(path, 'w') as f:
        f.write("""import numpy as np
from tinygrad.tensor import Tensor

def to_tensor(x):
    \"\"\"Convert numpy array to TinyGrad tensor\"\"\"
    if isinstance(x, Tensor):
        return x
    return Tensor(x)

def to_numpy(x):
    \"\"\"Convert TinyGrad tensor to numpy array\"\"\"
    if isinstance(x, Tensor):
        return x.numpy()
    return x
""")

def create_agent_file(path):
    with open(path, 'w') as f:
        f.write("""import numpy as np
from tinygrad.tensor import Tensor
import random
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import RLCVModel

class RLAgent:
    \"\"\"Base class for reinforcement learning agents\"\"\"
    
    def __init__(self, state_shape, action_space, config=None):
        \"\"\"
        Initialize the RL agent
        
        Args:
            state_shape: Shape of the state input
            action_space: Number of possible actions or action space
            config: Configuration dictionary
        \"\"\"
        self.state_shape = state_shape
        self.action_space = action_space
        self.config = config or {}
        
        # Default configuration
        self.gamma = self.config.get('gamma', 0.99)  # Discount factor
        self.epsilon = self.config.get('epsilon', 1.0)  # Exploration rate
        self.epsilon_min = self.config.get('epsilon_min', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        
        # To be initialized in subclasses
        self.model = None
        
    def build_model(self):
        \"\"\"Build the model - to be implemented by subclasses\"\"\"
        raise NotImplementedError
        
    def act(self, state):
        \"\"\"Choose an action based on the current state\"\"\"
        raise NotImplementedError
        
    def train(self, state, action, reward, next_state, done):
        \"\"\"Train the agent on a single experience tuple\"\"\"
        raise NotImplementedError


class DQNAgent(RLAgent):
    \"\"\"Deep Q-Network agent implementation using TinyGrad\"\"\"
    
    def build_model(self):
        \"\"\"Build a Q-network model using TinyGrad\"\"\"
        # Create a CNN model for Q-value prediction
        self.model = RLCVModel(
            input_shape=self.state_shape,
            action_space=self.action_space,
            config=self.config
        )
        self.model.build()
        
        # Target network for stable learning
        self.target_model = RLCVModel(
            input_shape=self.state_shape,
            action_space=self.action_space,
            config=self.config
        )
        self.target_model.build()
        
        # Update frequency for target network
        self.target_update_freq = self.config.get('target_update_freq', 100)
        self.update_counter = 0
        
    def act(self, state):
        \"\"\"Choose an action using epsilon-greedy policy\"\"\"
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        
        # Convert state to proper format
        if isinstance(state, np.ndarray):
            # Add batch dimension if needed
            if len(state.shape) == 3:  # (C, H, W)
                state = np.expand_dims(state, axis=0)  # (1, C, H, W)
            state = Tensor(state)
        
        # Get Q-values and choose the best action
        q_values = self.model.forward(state)
        q_values_np = q_values.numpy()
        return np.argmax(q_values_np[0])
    
    def train(self, states, actions, rewards, next_states, dones):
        \"\"\"Train the agent on a batch of experiences\"\"\"
        # Simple implementation for testing
        # In a real implementation, this would update the model weights
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            # In a real implementation, this would copy weights
            pass
            
        return True
        
    def save(self, path):
        \"\"\"Save the model to disk\"\"\"
        # In a real implementation, this would save the model weights
        print(f"Model would be saved to {path}")
""")

def create_environment_file(path):
    with open(path, 'w') as f:
        f.write("""import numpy as np

class Environment:
    \"\"\"Base class for all environments\"\"\"
    
    def __init__(self):
        \"\"\"Initialize the environment\"\"\"
        pass
        
    def reset(self):
        \"\"\"Reset the environment and return the initial state\"\"\"
        raise NotImplementedError
        
    def step(self, action):
        \"\"\"
        Take an action in the environment
        
        Args:
            action: The action to take
            
        Returns:
            next_state: The next state
            reward: The reward received
            done: Whether the episode is done
            info: Additional information
        \"\"\"
        raise NotImplementedError
        
    def render(self):
        \"\"\"Render the environment\"\"\"
        pass


class VisionEnvironment(Environment):
    \"\"\"Base class for vision-based environments\"\"\"
    
    def __init__(self, preprocessor=None):
        \"\"\"
        Initialize the vision environment
        
        Args:
            preprocessor: Image preprocessor
        \"\"\"
        super().__init__()
        self.preprocessor = preprocessor
        
    def get_observation(self):
        \"\"\"Get the current observation (image)\"\"\"
        raise NotImplementedError
        
    def process_observation(self, observation):
        \"\"\"Process the observation using the preprocessor\"\"\"
        if self.preprocessor is not None:
            return self.preprocessor.process(observation)
        return observation


class ObjectTrackingEnv(VisionEnvironment):
    \"\"\"Environment for object tracking tasks\"\"\"
    
    def __init__(self, video_source=None, preprocessor=None, target_object=None):
        \"\"\"
        Initialize the object tracking environment
        
        Args:
            video_source: Source of video frames
            preprocessor: Image preprocessor
            target_object: Target object to track
        \"\"\"
        super().__init__(preprocessor)
        self.video_source = video_source
        self.target_object = target_object
        self.current_frame = 0
        self.max_frames = 100  # For testing
        
    def reset(self):
        \"\"\"Reset the environment and return the initial state\"\"\"
        self.current_frame = 0
        # Create a random initial state
        if self.preprocessor:
            return self.preprocessor.process(None)
        return np.random.rand(3, 64, 64).astype(np.float32)
        
    def step(self, action):
        \"\"\"Take an action in the environment\"\"\"
        self.current_frame += 1
        done = self.current_frame >= self.max_frames
        
        # Generate a random next state
        if self.preprocessor:
            next_state = self.preprocessor.process(None)
        else:
            next_state = np.random.rand(3, 64, 64).astype(np.float32)
            
        # Simple reward function for testing
        reward = np.random.uniform(-1, 1)
        
        # Additional info
        info = {"frame": self.current_frame}
        
        return next_state, reward, done, info
        
    def render(self):
        \"\"\"Render the environment\"\"\"
        print(f"Rendering frame {self.current_frame}")
""")

def create_memory_file(path):
    with open(path, 'w') as f:
        f.write("""import numpy as np
import random
from collections import deque

class ReplayMemory:
    \"\"\"Simple replay memory for RL agents\"\"\"
    
    def __init__(self, capacity=10000):
        \"\"\"
        Initialize the replay memory
        
        Args:
            capacity: Maximum number of experiences to store
        \"\"\"
        self.memory = deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):
        \"\"\"Add an experience to memory\"\"\"
        self.memory.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        \"\"\"Sample a batch of experiences from memory\"\"\"
        return random.sample(self.memory, min(len(self.memory), batch_size))
        
    def __len__(self):
        \"\"\"Return the current size of memory\"\"\"
        return len(self.memory)


class PrioritizedReplayMemory:
    \"\"\"Prioritized replay memory for RL agents\"\"\"
    
    def __init__(self, capacity=10000, alpha=0.6, beta=0.4, beta_increment=0.001):
        \"\"\"
        Initialize the prioritized replay memory
        
        Args:
            capacity: Maximum number of experiences to store
            alpha: How much prioritization to use (0 = no prioritization, 1 = full prioritization)
            beta: Importance sampling factor (0 = no correction, 1 = full correction)
            beta_increment: How much to increase beta each time we sample
        \"\"\"
        self.memory = []
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0
        self.max_priority = 1.0
        
    def add(self, state, action, reward, next_state, done):
        \"\"\"Add an experience to memory\"\"\"
        # For simplicity in testing, use max priority for new experiences
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        if self.size < self.capacity:
            self.memory.append((state, action, reward, next_state, done))
            self.size += 1
        else:
            self.memory[self.position] = (state, action, reward, next_state, done)
            
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        \"\"\"Sample a batch of experiences based on priority\"\"\"
        if len(self.memory) < batch_size:
            return [], [], []
            
        # Increase beta over time for more accurate bias correction
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.memory)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (len(self.memory) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Get experiences
        experiences = [self.memory[idx] for idx in indices]
        
        return experiences, indices, weights
    
    def update_priorities(self, indices, priorities):
        \"\"\"Update priorities for sampled experiences\"\"\"
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        \"\"\"Return the current size of memory\"\"\"
        return len(self.memory)
""")

def create_policy_file(path):
    with open(path, 'w') as f:
        f.write("""import numpy as np
import random

class Policy:
    \"\"\"Base class for all policies\"\"\"
    
    def __init__(self):
        \"\"\"Initialize the policy\"\"\"
        pass
        
    def select_action(self, state, **kwargs):
        \"\"\"Select an action based on the current state\"\"\"
        raise NotImplementedError


class EpsilonGreedyPolicy(Policy):
    \"\"\"Epsilon-greedy policy for action selection\"\"\"
    
    def __init__(self, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        \"\"\"
        Initialize the epsilon-greedy policy
        
        Args:
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
        \"\"\"
        super().__init__()
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
    def select_action(self, state, action_values=None, action_space=None):
        \"\"\"
        Select an action using epsilon-greedy strategy
        
        Args:
            state: Current state
            action_values: Q-values for each action
            action_space: Number of possible actions
            
        Returns:
            Selected action
        \"\"\"
        if action_values is None or np.random.rand() <= self.epsilon:
            # Exploration: random action
            return random.randrange(action_space)
        
        # Exploitation: best action
        return np.argmax(action_values)
        
    def update(self):
        \"\"\"Update epsilon value\"\"\"
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return self.epsilon


class BoltzmannPolicy(Policy):
    \"\"\"Boltzmann (softmax) policy for action selection\"\"\"
    
    def __init__(self, temperature=1.0, temperature_min=0.1, temperature_decay=0.995):
        \"\"\"
        Initialize the Boltzmann policy
        
        Args:
            temperature: Temperature parameter for exploration
            temperature_min: Minimum temperature
            temperature_decay: Decay rate for temperature
        \"\"\"
        super().__init__()
        self.temperature = temperature
        self.temperature_min = temperature_min
        self.temperature_decay = temperature_decay
        
    def select_action(self, state, action_values=None, action_space=None):
        \"\"\"
        Select an action using Boltzmann distribution
        
        Args:
            state: Current state
            action_values: Q-values for each action
            action_space: Number of possible actions
            
        Returns:
            Selected action
        \"\"\"
        if action_values is None:
            # If no Q-values, use uniform random
            return random.randrange(action_space)
        
        # Apply temperature scaling
        scaled_values = action_values / self.temperature
        
        # Compute softmax probabilities
        exp_values = np.exp(scaled_values - np.max(scaled_values))  # Subtract max for numerical stability
        probabilities = exp_values / np.sum(exp_values)
        
        # Sample action based on probabilities
        return np.random.choice(len(action_values), p=probabilities)
        
    def update(self):
        \"\"\"Update temperature value\"\"\"
        if self.temperature > self.temperature_min:
            self.temperature *= self.temperature_decay
        return self.temperature
""")

def create_preprocessing_file(path):
    with open(path, 'w') as f:
        f.write("""import numpy as np

class ImagePreprocessor:
    \"\"\"Image preprocessing utility for RL computer vision tasks\"\"\"
    
    def __init__(self, target_size=(64, 64), normalize=True, grayscale=False):
        \"\"\"
        Initialize the image preprocessor
        
        Args:
            target_size: Target image size (height, width)
            normalize: Whether to normalize pixel values to [0, 1]
            grayscale: Whether to convert to grayscale
        \"\"\"
        self.target_size = target_size
        self.normalize = normalize
        self.grayscale = grayscale
        
    def process(self, image):
        \"\"\"
        Process an image
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Processed image
        \"\"\"
        # For testing, just create a random image with the right shape
        if image is None:
            channels = 1 if self.grayscale else 3
            return np.random.rand(channels, *self.target_size).astype(np.float32)
            
        # In a real implementation, this would resize, normalize, etc.
        # For now, just return a placeholder
        return np.random.rand(3, *self.target_size).astype(np.float32)


class FeatureExtractor:
    \"\"\"Feature extraction utility for computer vision tasks\"\"\"
    
    def __init__(self, feature_dim=64, use_pretrained=False):
        \"\"\"
        Initialize the feature extractor
        
        Args:
            feature_dim: Dimension of extracted features
            use_pretrained: Whether to use a pretrained model
        \"\"\"
        self.feature_dim = feature_dim
        self.use_pretrained = use_pretrained
        
    def extract(self, image):
        \"\"\"
        Extract features from an image
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Extracted features
        \"\"\"
        # For testing, just create random features
        return np.random.rand(self.feature_dim).astype(np.float32)
""")

def main():
    """Main function to set up the project"""
    print("Setting up TinyGrad RLCV project...")
    
    # Create directory structure
    create_directory_structure()
    
    # Check and create required files
    check_required_files()
    
    print("Project setup complete!")
    print("You can now run the examples or tests.")

if __name__ == "__main__":
    main()