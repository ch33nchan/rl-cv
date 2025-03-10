import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.agent import DQNAgent
from rl.environment import ObjectTrackingEnv
from rl.memory import ReplayMemory
from vision.preprocessing import ImagePreprocessor

def run_simple_tracking():
    """Run a simple object tracking example"""
    print("Running simple object tracking example...")
    

    preprocessor = ImagePreprocessor(
        target_size=(32, 32),  
        normalize=True,
        grayscale=False
    )
    
  
    env = ObjectTrackingEnv(
        video_source=None, 
        preprocessor=preprocessor,
        target_object=None  
    )
    
    
    initial_state = env.reset()
    state_shape = initial_state.shape
    print(f"State shape: {state_shape}")
    
   
    agent = DQNAgent(
        state_shape=state_shape,
        action_space=9,  
        config={
            'gamma': 0.99,
            'epsilon': 1.0,
            'epsilon_min': 0.1,
            'epsilon_decay': 0.995,
            'learning_rate': 0.001,
            'target_update_freq': 10,
            'filters': [8, 16],  
            'kernel_sizes': [3, 3],
            'strides': [1, 2],
            'fc_units': [64]
        }
    )
    

    agent.build_model()
    
    
    memory = ReplayMemory(capacity=1000)

    total_rewards = []
    episodes = 10  
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        while not done:
           
            action = agent.act(state)
            
           
            next_state, reward, done, _ = env.step(action)
            
           
            memory.add(state, action, reward, next_state, done)
            
            
            state = next_state
            total_reward += reward
            step += 1
            
            
            if len(memory) >= 32:  
                experiences = memory.sample(32)
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
            
       
            if step >= 100:
                done = True
        

        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards)
        
        print(f"Episode {episode+1}/{episodes}, Steps: {step}, Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}")
    
   
    plt.figure(figsize=(10, 5))
    plt.plot(total_rewards)
    plt.title('Training Progress - Object Tracking')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig('tracking_progress.png')
    plt.close()
    
    print("Example completed!")

if __name__ == "__main__":
    run_simple_tracking()