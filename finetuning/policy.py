import numpy as np
from tinygrad.tensor import Tensor

class Policy:

    
    def __init__(self):

        pass
        
    def select_action(self, state):

        raise NotImplementedError


class EpsilonGreedyPolicy(Policy):
   
    
    def __init__(self, action_space, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):

        super().__init__()
        self.action_space = action_space
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
    def select_action(self, q_values):

        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space)
        

        if isinstance(q_values, Tensor):
            q_values = q_values.numpy()
            
        return np.argmax(q_values)
    
    def decay_epsilon(self):

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def reset(self):

        self.epsilon = 1.0


class SoftmaxPolicy(Policy):

    
    def __init__(self, temperature=1.0):
    
        super().__init__()
        self.temperature = temperature
        
    def select_action(self, q_values):

        if isinstance(q_values, Tensor):
            q_values = q_values.numpy()
            

        scaled_q = q_values / self.temperature
        

        exp_q = np.exp(scaled_q - np.max(scaled_q))
        probabilities = exp_q / np.sum(exp_q)
        

        return np.random.choice(len(q_values), p=probabilities)
    
    def set_temperature(self, temperature):
        
        self.temperature = temperature