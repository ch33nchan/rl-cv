import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tinygrad"))
from tinygrad.tensor import Tensor

import random
import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import RLCVModel

class DQNAgent:

    
    def __init__(self, state_shape, action_space, config=None):

        self.state_shape = state_shape
        self.action_space = action_space
        self.config = config or {}
        
       
        self.gamma = self.config.get('gamma', 0.99)  
        self.epsilon = self.config.get('epsilon', 1.0) 
        self.epsilon_min = self.config.get('epsilon_min', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        
 
        self.model = None
        self.target_model = None
        
    def build_model(self):

        self.model = RLCVModel(
            input_shape=self.state_shape,
            action_space=self.action_space,
            config=self.config
        )
        self.model.build()

        self.target_model = RLCVModel(
            input_shape=self.state_shape,
            action_space=self.action_space,
            config=self.config
        )
        self.target_model.build()
  
        self.target_update_freq = self.config.get('target_update_freq', 100)
        self.update_counter = 0
        
    def act(self, state):
       
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)

        if isinstance(state, np.ndarray):
           
            if len(state.shape) == 3: 
                state = np.expand_dims(state, axis=0)
            state = Tensor(state)
        
        q_values = self.model.forward(state)
        q_values_np = q_values.numpy()
        return np.argmax(q_values_np[0])
    
    def train(self, states, actions, rewards, next_states, dones):

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
        
            pass
            
        return True
        
    def save(self, path):
        print(f"Model would be saved to {path}")