import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try importing tinygrad, install if not available
try:
    from tinygrad.tensor import Tensor
except ImportError:
    print("TinyGrad not found. Attempting to install...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tinygrad"])
    from tinygrad.tensor import Tensor

try:
    from core.model import RLCVModel
    from core.tinygrad_utils import to_tensor, to_numpy
    from rl.agent import DQNAgent
    from rl.environment import ObjectTrackingEnv
    from vision.preprocessing import ImagePreprocessor
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Make sure all required modules are created in the project structure.")
    sys.exit(1)

def test_model_forward():
    """Test basic model forward pass with minimal compute"""
    print("Testing model forward pass...")
    
    # Create a tiny input (1x3x16x16)
    input_shape = (3, 16, 16)
    x = np.random.rand(1, *input_shape).astype(np.float32)
    x_tensor = to_tensor(x)
    
    # Create a minimal model
    model = RLCVModel(
        input_shape=input_shape,
        action_space=4,  # Minimal action space
        config={
            'filters': [4, 8],  # Tiny filters
            'kernel_sizes': [3, 3],
            'strides': [1, 2],
            'fc_units': [16]  # Small FC layer
        }
    )
    
    # Build and run forward pass
    model.build()
    output = model.forward(x_tensor)
    
    # Convert to numpy and print shape
    output_np = to_numpy(output)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_np.shape}")
    print(f"Output values: {output_np}")
    
    return True

def test_agent_act():
    """Test agent action selection with minimal compute"""
    print("\nTesting agent action selection...")
    
    # Create a tiny state (3x16x16)
    state_shape = (3, 16, 16)
    state = np.random.rand(*state_shape).astype(np.float32)
    
    # Create a minimal agent
    agent = DQNAgent(
        state_shape=state_shape,
        action_space=4,
        config={
            'gamma': 0.99,
            'epsilon': 0.5,  # 50% exploration
            'filters': [4, 8],  # Tiny filters
            'kernel_sizes': [3, 3],
            'strides': [1, 2],
            'fc_units': [16]  # Small FC layer
        }
    )
    
    # Build model and select action
    agent.build_model()
    action = agent.act(state)
    
    print(f"State shape: {state.shape}")
    print(f"Selected action: {action}")
    
    return True

def test_environment_step():
    """Test environment step with minimal compute"""
    print("\nTesting environment step...")
    
    # Create a minimal preprocessor
    preprocessor = ImagePreprocessor(
        target_size=(16, 16),  # Tiny images
        normalize=True,
        grayscale=False
    )
    
    # Create a minimal environment
    env = ObjectTrackingEnv(
        video_source=None,  # Using placeholder frames
        preprocessor=preprocessor,
        target_object=None  # Using placeholder detection
    )
    
    # Reset and take a step
    state = env.reset()
    action = 0  # Just use action 0
    next_state, reward, done, info = env.step(action)
    
    print(f"State shape: {state.shape}")
    print(f"Next state shape: {next_state.shape}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    
    return True

def test_minimal_training_loop():
    """Test a minimal training loop"""
    print("\nTesting minimal training loop...")
    
    # Create a minimal preprocessor
    preprocessor = ImagePreprocessor(
        target_size=(16, 16),  # Tiny images
        normalize=True,
        grayscale=False
    )
    
    # Create a minimal environment
    env = ObjectTrackingEnv(
        video_source=None,  # Using placeholder frames
        preprocessor=preprocessor,
        target_object=None  # Using placeholder detection
    )
    
    # Reset to get state shape
    state = env.reset()
    state_shape = state.shape
    
    # Create a minimal agent
    agent = DQNAgent(
        state_shape=state_shape,
        action_space=4,
        config={
            'gamma': 0.99,
            'epsilon': 0.5,
            'filters': [4, 8],
            'kernel_sizes': [3, 3],
            'strides': [1, 2],
            'fc_units': [16]
        }
    )
    
    # Build model
    agent.build_model()
    
    # Run a few steps
    total_reward = 0
    for step in range(10):  # Just 10 steps
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        
        # Manually create a batch of 1 for training
        agent.train([state], [action], [reward], [next_state], [done])
        
        state = next_state
        total_reward += reward
        
        print(f"Step {step+1}, Action: {action}, Reward: {reward:.2f}")
        
        if done:
            break
    
    print(f"Total steps: {step+1}, Total reward: {total_reward:.2f}")
    return True

if __name__ == "__main__":
    print("Running minimal tests for TinyGrad RLCV...")
    
    try:
        test_model_forward()
        test_agent_act()
        test_environment_step()
        test_minimal_training_loop()
        print("\nAll tests completed successfully!")
    except Exception as e:
        print(f"\nTest failed with error: {e}")