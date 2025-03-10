import numpy as np
import time
import os
import argparse

from rl.agent import DQNAgent
from rl.environment import ObjectTrackingEnv, SimpleDetectionEnv
from rl.memory import ReplayMemory, PrioritizedReplayMemory
from vision.preprocessing import ImagePreprocessor

def parse_args():
    parser = argparse.ArgumentParser(description='Train RLCV model')
    parser.add_argument('--task', type=str, default='tracking', choices=['tracking', 'detection'],
                        help='Task to train on (tracking or detection)')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Initial exploration rate')
    parser.add_argument('--epsilon-min', type=float, default=0.01, help='Minimum exploration rate')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Exploration decay rate')
    parser.add_argument('--memory-size', type=int, default=10000, help='Replay memory size')
    parser.add_argument('--target-update', type=int, default=100, help='Target network update frequency')
    parser.add_argument('--save-dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--render', action='store_true', help='Render environment')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create preprocessor
    preprocessor = ImagePreprocessor(
        target_size=(64, 64),
        normalize=True,
        grayscale=False
    )
    
    # Create environment based on task
    if args.task == 'tracking':
        env = ObjectTrackingEnv(
            video_source=None,  # Using placeholder frames
            preprocessor=preprocessor,
            target_object=None  # Using placeholder detection
        )
        action_space = 9  # 8 directions + stay
    else:  # detection
        env = SimpleDetectionEnv(
            image_source=None,  # Using placeholder images
            preprocessor=preprocessor,
            target_classes=['person', 'car', 'dog']  # Example classes
        )
        action_space = 10  # Example output size for detection
    
    # Get initial state shape
    initial_state = env.reset()
    state_shape = initial_state.shape
    print(f"State shape: {state_shape}")
    
    # Create agent
    agent = DQNAgent(
        state_shape=state_shape,
        action_space=action_space,
        config={
            'gamma': args.gamma,
            'epsilon': args.epsilon,
            'epsilon_min': args.epsilon_min,
            'epsilon_decay': args.epsilon_decay,
            'learning_rate': args.lr,
            'target_update_freq': args.target_update,
            'filters': [16, 32, 32],  # CNN architecture
            'kernel_sizes': [3, 3, 3],
            'strides': [1, 2, 2],
            'fc_units': [256]
        }
    )
    
    # Build agent model
    agent.build_model()
    
    # Create memory
    memory = ReplayMemory(capacity=args.memory_size)
    
    # Training loop
    total_rewards = []
    
    for episode in range(args.episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        while not done:
            # Select action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Store experience
            memory.add(state, action, reward, next_state, done)
            
            # Update state and reward
            state = next_state
            total_reward += reward
            step += 1
            
            # Train agent
            if len(memory) >= args.batch_size:
                experiences = memory.sample(args.batch_size)
                states_batch = []
                actions_batch = []
                rewards_batch = []
                next_states_batch = []
                dones_batch = []
                
                for exp in experiences:
                    s, a, r, ns, d = exp
                    states_batch.append(s)
                    actions_batch.append(a)
                    rewards_batch.append(r)
                    next_states_batch.append(ns)
                    dones_batch.append(d)
                
                agent.train(states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch)
            
            # Render if requested
            if args.render:
                env.render()
                time.sleep(0.01)
        
        # End of episode
        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards[-100:]) if len(total_rewards) >= 100 else np.mean(total_rewards)
        
        print(f"Episode {episode+1}/{args.episodes}, Steps: {step}, Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
        
        # Save model periodically
        if (episode + 1) % 100 == 0:
            save_path = os.path.join(args.save_dir, f"{args.task}_model_ep{episode+1}.pt")
            agent.save(save_path)
            print(f"Model saved to {save_path}")
    
    # Save final model
    final_save_path = os.path.join(args.save_dir, f"{args.task}_model_final.pt")
    agent.save(final_save_path)
    print(f"Final model saved to {final_save_path}")
    
    # Plot training progress
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(total_rewards)
        plt.title(f'Training Progress - {args.task.capitalize()}')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        plt.savefig(os.path.join(args.save_dir, f"{args.task}_training_progress.png"))
        plt.close()
    except ImportError:
        print("Matplotlib not available, skipping plot generation")

if __name__ == "__main__":
    main()