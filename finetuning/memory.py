import numpy as np
import random
from collections import deque

class ReplayMemory:

    
    def __init__(self, capacity=10000):

        self.memory = deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):

        self.memory.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):

        return random.sample(self.memory, min(len(self.memory), batch_size))
        
    def __len__(self):

        return len(self.memory)


class PrioritizedReplayMemory:

    
    def __init__(self, capacity=10000, alpha=0.6, beta=0.4, beta_increment=0.001):

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

        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        if self.size < self.capacity:
            self.memory.append((state, action, reward, next_state, done))
            self.size += 1
        else:
            self.memory[self.position] = (state, action, reward, next_state, done)
            
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
       
        if len(self.memory) < batch_size:
            return [], [], []
            

        self.beta = min(1.0, self.beta + self.beta_increment)
        
    
        priorities = self.priorities[:len(self.memory)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
       
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        
        
        weights = (len(self.memory) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
       
        experiences = [self.memory[idx] for idx in indices]
        
        return experiences, indices, weights
    
    def update_priorities(self, indices, priorities):

        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):

        return len(self.memory)

