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

class WebcamTrackingEnv(ObjectTrackingEnv):
   
    
    def __init__(self, preprocessor=None, target_object=None, camera_id=0):
        
        super().__init__(video_source=None, preprocessor=preprocessor, target_object=target_object)
        self.camera_id = camera_id
        self.cap = None
        self.current_frame_raw = None
        self.bbox = [100, 100, 200, 200]  # Initial bounding box [x, y, w, h]
        
    def _setup_camera(self):
        
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera with ID {self.camera_id}")
    
    def reset(self):
        
        self._setup_camera()
        self.current_frame = 0
        
        
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera")
        
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.current_frame_raw = frame
        
       
        h, w = frame.shape[:2]
        self.bbox = [w//2-50, h//2-50, 100, 100]  # [x, y, w, h]
        
       
        if self.preprocessor:
            return self.preprocessor.process(frame)
        return frame
    
    def step(self, action):
        """Take an action in the environment"""
        start_time = time.time()
        self.current_frame += 1
        
        
        ret, frame = self.cap.read()
        if not ret:
            return self.reset(), 0, True, {"frame": self.current_frame}
        
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.current_frame_raw = frame
        
        
        move_size = 20  
        
        
        random_factor = np.random.uniform(0.8, 1.2)
        move_size = int(move_size * random_factor)
        
        dx, dy = 0, 0
        
        if action == 1: 
            dy = -move_size
        elif action == 2:  
            dx, dy = move_size, -move_size
        elif action == 3: 
            dx = move_size
        elif action == 4: 
            dx, dy = move_size, move_size
        elif action == 5:  
            dy = move_size
        elif action == 6:  
            dx, dy = -move_size, move_size
        elif action == 7:  
            dx = -move_size
        elif action == 8:  
            dx, dy = -move_size, -move_size
        
       
        self.bbox[0] = max(0, min(frame.shape[1] - self.bbox[2], self.bbox[0] + dx))
        self.bbox[1] = max(0, min(frame.shape[0] - self.bbox[3], self.bbox[1] + dy))
        
    
        x, y, w, h = self.bbox
        roi = frame[y:y+h, x:x+w]
        
        
        if roi.size > 0:
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
            mask1 = cv2.inRange(hsv_roi, np.array([0, 100, 100]), np.array([10, 255, 255]))
            mask2 = cv2.inRange(hsv_roi, np.array([160, 100, 100]), np.array([180, 255, 255]))
            red_pixels = cv2.countNonZero(mask1 + mask2)
            reward = red_pixels / (w * h) - 0.5  # Normalize and center around 0
        else:
            reward = -1.0
        
    
        if self.preprocessor:
            next_state = self.preprocessor.process(frame)
        else:
            next_state = frame
        
        done = self.current_frame >= self.max_frames
        
  
        processing_time = time.time() - start_time
        
       
        cpu_percent = psutil.cpu_percent()
 
        memory_info = psutil.Process(os.getpid()).memory_info()
        memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
        
        info = {
            "frame": self.current_frame, 
            "bbox": self.bbox,
            "processing_time": processing_time,
            "cpu_percent": cpu_percent,
            "memory_mb": memory_mb,
            "action": action,
            "reward": reward
        }
        
        return next_state, reward, done, info
    
    def render(self, mode='human'):
        """Render the environment"""
        if self.current_frame_raw is None:
            return
        
       
        frame = self.current_frame_raw.copy()
        x, y, w, h = self.bbox
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
       
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)  
        line_type = 1
        
     
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        cv2.putText(frame, f"Time: {timestamp}", (10, 20), font, font_scale, font_color, line_type)
        
        
        cv2.putText(frame, f"Frame: {self.current_frame}", (10, 40), font, font_scale, font_color, line_type)
      
        if hasattr(self, 'last_info') and 'cpu_percent' in self.last_info:
            cv2.putText(frame, f"CPU: {self.last_info['cpu_percent']:.1f}%", (10, 60), font, font_scale, font_color, line_type)
        
        
        if hasattr(self, 'last_info') and 'memory_mb' in self.last_info:
            cv2.putText(frame, f"Memory: {self.last_info['memory_mb']:.1f} MB", (10, 80), font, font_scale, font_color, line_type)
        
        
        if hasattr(self, 'last_info') and 'processing_time' in self.last_info:
            fps = 1.0 / self.last_info['processing_time'] if self.last_info['processing_time'] > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 100), font, font_scale, font_color, line_type)
        
    
        if hasattr(self, 'last_info') and 'action' in self.last_info:
            action_names = ["Stay", "Up", "Up-Right", "Right", "Down-Right", "Down", "Down-Left", "Left", "Up-Left"]
            action_name = action_names[self.last_info['action']] if 0 <= self.last_info['action'] < len(action_names) else str(self.last_info['action'])
            cv2.putText(frame, f"Action: {action_name}", (10, 120), font, font_scale, font_color, line_type)
        
        if hasattr(self, 'last_info') and 'reward' in self.last_info:
            cv2.putText(frame, f"Reward: {self.last_info['reward']:.2f}", (10, 140), font, font_scale, font_color, line_type)
        
        
        cv2.imshow('Object Tracking', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
    
    def close(self):
       
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

def run_webcam_tracking():
   
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

        env.close()
    
    print("Webcam tracking completed!")

if __name__ == "__main__":
    run_webcam_tracking()