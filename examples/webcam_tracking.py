import numpy as np
import os
import sys
import time
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import psutil  
import datetime


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.agent import DQNAgent
from rl.environment import ObjectTrackingEnv
from rl.memory import ReplayMemory
from vision.preprocessing import ImagePreprocessor
import tkinter as tk
from tkinter import colorchooser
import colorsys

class WebcamTrackingEnv(ObjectTrackingEnv):
    def __init__(self, preprocessor=None, target_object=None, camera_id=0):
        super().__init__(video_source=None, preprocessor=preprocessor, target_object=target_object)
        self.camera_id = camera_id
        self.cap = None
        self.current_frame_raw = None
        self.bbox = [100, 100, 200, 200]
        self.color_filter = None
        self.filter_threshold = 30
        self.current_frame = 0
        
    def reset(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.camera_id)
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Failed to capture frame from camera")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.current_frame_raw = frame
        if self.preprocessor:
            return self.preprocessor.process(frame)
        return frame

    def step(self, action):
        start_time = time.time()
        self.current_frame += 1
        
        ret, frame = self.cap.read()
        if not ret:
            return self.reset(), 0, True, {"frame": self.current_frame}
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.current_frame_raw = frame
        
        # Create info dict first
        info = {
            "frame": self.current_frame,
            "bbox": self.bbox,
            "processing_time": 0,  # Will update later
            "cpu_percent": 0,      # Will update later
            "memory_mb": 0         # Will update later
        }
        
        # Apply color filter before processing
        if self.color_filter is not None:
            filtered_frame = self.apply_color_filter(frame)
            # Calculate percentage of filtered pixels
            non_zero = np.count_nonzero(np.any(filtered_frame > 0, axis=2))
            total_pixels = frame.shape[0] * frame.shape[1]
            filter_ratio = non_zero / total_pixels
            info["filter_ratio"] = filter_ratio
            # Adjust reward based on filtered pixels
            if filter_ratio > 0.01:  # If more than 1% of pixels match filter
                reward = 1.5  # Increased reward for matching color
            else:
                reward = 1.0  # Base reward
        else:
            reward = 1.0  # Base reward when no filter is active
        
        # Update bounding box based on action
        move_size = 20
        if action == 1:  # up
            self.bbox[1] = max(0, self.bbox[1] - move_size)
        elif action == 2:  # right
            self.bbox[0] = min(frame.shape[1] - self.bbox[2], self.bbox[0] + move_size)
        elif action == 3:  # down
            self.bbox[1] = min(frame.shape[0] - self.bbox[3], self.bbox[1] + move_size)
        elif action == 4:  # left
            self.bbox[0] = max(0, self.bbox[0] - move_size)
        
        # Update performance metrics
        processing_time = time.time() - start_time
        info["processing_time"] = processing_time
        info["cpu_percent"] = psutil.cpu_percent()
        info["memory_mb"] = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Prepare next state
        if self.preprocessor:
            next_state = self.preprocessor.process(frame)
        else:
            next_state = frame
        
        # Adjust reward based on bbox position
        if not (0 <= self.bbox[0] <= frame.shape[1] - self.bbox[2] and 
                0 <= self.bbox[1] <= frame.shape[0] - self.bbox[3]):
            reward = -1.0
        
        return next_state, reward, False, info

    def apply_color_filter(self, frame):
        """Apply color filter to frame"""
        if self.color_filter is None:
            return frame
            
        # Convert RGB to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        rgb = np.uint8([[self.color_filter]])
        hsv_color = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h, s, v = hsv_color[0][0]
        
        # Create mask with safe HSV ranges
        lower = np.array([max(0, h - 10), max(0, s - 50), max(0, v - 50)])
        upper = np.array([min(180, h + 10), min(255, s + 50), min(255, v + 50)])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Apply mask to frame
        result = cv2.bitwise_and(frame, frame, mask=mask)
        return result
    
    def render(self, mode='human'):
        if self.current_frame_raw is None:
            return
            
        frame = self.current_frame_raw.copy()
        
        # Apply color filter if set
        if self.color_filter is not None:
            frame = self.apply_color_filter(frame)
            color_text = f"Filter: RGB{self.color_filter}"
            cv2.putText(frame, color_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
            if hasattr(self, 'last_info') and 'filter_ratio' in self.last_info:
                ratio_text = f"Match: {self.last_info['filter_ratio']*100:.1f}%"
                cv2.putText(frame, ratio_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (255, 255, 255), 2)
        
        # Draw bounding box
        x, y, w, h = self.bbox
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Object Tracking', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF
        
        # Handle key events
        if key == ord('q'):
            cv2.destroyAllWindows()
            if self.cap is not None:
                self.cap.release()
            sys.exit(0)
        elif key == ord('f'):
            self.color_filter = (255, 0, 0)  # Red
        elif key == ord('g'):
            self.color_filter = (0, 255, 0)  # Green
        elif key == ord('b'):
            self.color_filter = (0, 0, 255)  # Blue
        elif key == ord('y'):
            self.color_filter = (255, 255, 0)  # Yellow
        elif key == ord('m'):
            self.color_filter = (255, 0, 255)  # Magenta
        elif key == ord('c'):
            self.color_filter = None  # Clear filter
    
    def open_color_picker(self):
        """Open color picker window"""
        # Initialize root with required attributes for macOS
        root = tk.Tk()
        root.withdraw()
        root.update_idletasks()
        root.lift()
        root.attributes('-topmost', True)
        
        try:
            color = colorchooser.askcolor(title="Choose color filter")
            if color[1]:  # If color was selected (not cancelled)
                self.set_color_filter(color[1])
        finally:
            root.destroy()

def run_webcam_tracking():
    """Run webcam tracking with color filtering"""
    print("Running webcam tracking with color filtering...")
    print("Controls:")
    print("  'f' - Set color filter")
    print("  'c' - Clear color filter")
    print("  'q' - Quit")
    
    
    print("Running webcam object tracking...")
    

    preprocessor = ImagePreprocessor(
        target_size=(64, 64),
        normalize=True,
        grayscale=False
    )
    
   
    env = WebcamTrackingEnv(
        preprocessor=preprocessor,
        target_object=None,  
        camera_id=0  
    )
    
    try:
      
        initial_state = env.reset()
        state_shape = initial_state.shape
        print(f"State shape: {state_shape}")
        
       
        agent = DQNAgent(
            state_shape=state_shape,
            action_space=9,  
            config={
                'gamma': 0.99,
                'epsilon': 0.3, 
                'epsilon_min': 0.1,
                'epsilon_decay': 0.999,
                'learning_rate': 0.001,
                'target_update_freq': 10,
                'filters': [16, 32, 64],
                'kernel_sizes': [3, 3, 3],
                'strides': [1, 2, 2],
                'fc_units': [128]
            }
        )

        agent.build_model()
        
    
        memory = ReplayMemory(capacity=5000)
        

        total_rewards = []
        episodes = 5  
        max_steps = 1000 
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            step = 0
            
            while not done and step < max_steps:
                
                action = agent.act(state)
                
               
                next_state, reward, done, info = env.step(action)
                
                
                env.last_info = info
                
                
                env.render()
                
            
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
                
                
                if step % 10 == 0:
                    cpu_percent = info.get('cpu_percent', 0)
                    memory_mb = info.get('memory_mb', 0)
                    processing_time = info.get('processing_time', 0)
                    fps = 1.0 / processing_time if processing_time > 0 else 0
                    
                    print(f"Episode {episode+1}/{episodes}, Step {step}, Reward: {reward:.2f}, Total: {total_reward:.2f}")
                    print(f"  CPU: {cpu_percent:.1f}%, Memory: {memory_mb:.1f} MB, FPS: {fps:.1f}")
            
          
            total_rewards.append(total_reward)
            avg_reward = np.mean(total_rewards)
            
            print(f"Episode {episode+1}/{episodes}, Steps: {step}, Total Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}")
    
    finally:
        if env.cap is not None:
            env.cap.release()
        cv2.destroyAllWindows()
    
    print("Webcam tracking completed!")

    def close(self):
        """Close the environment and release resources"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam_tracking()