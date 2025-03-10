import numpy as np

class Environment:

    
    def __init__(self):

        pass
        
    def reset(self):

        raise NotImplementedError
        
    def step(self, action):
        raise NotImplementedError
        
    def render(self):

        pass


class ObjectTrackingEnv(Environment):

    
    def __init__(self, video_source=None, preprocessor=None, target_object=None):

        super().__init__()
        self.video_source = video_source
        self.preprocessor = preprocessor
        self.target_object = target_object
        self.current_frame = 0
        self.max_frames = 100  
        
    def reset(self):
        self.current_frame = 0
        # Create a random initial state
        if self.preprocessor:
            return self.preprocessor.process(None)
        return np.random.rand(3, 64, 64).astype(np.float32)
        
    def step(self, action):

        self.current_frame += 1
        done = self.current_frame >= self.max_frames

        if self.preprocessor:
            next_state = self.preprocessor.process(None)
        else:
            next_state = np.random.rand(3, 64, 64).astype(np.float32)
            

        reward = np.random.uniform(-1, 1)
        
        # Additional info
        info = {"frame": self.current_frame}
        
        return next_state, reward, done, info
        
    def render(self):

        print(f"Rendering frame {self.current_frame}")


class SimpleDetectionEnv(Environment):

    
    def __init__(self, image_source=None, preprocessor=None, target_classes=None):

        super().__init__()
        self.image_source = image_source
        self.preprocessor = preprocessor
        self.target_classes = target_classes or ["person", "car"]
        self.current_image = 0
        self.max_images = 100  
        
    def reset(self):

        self.current_image = 0

        if self.preprocessor:
            return self.preprocessor.process(None)
        return np.random.rand(3, 64, 64).astype(np.float32)
        
    def step(self, action):

        self.current_image += 1
        done = self.current_image >= self.max_images

        if self.preprocessor:
            next_state = self.preprocessor.process(None)
        else:
            next_state = np.random.rand(3, 64, 64).astype(np.float32)
            

        reward = np.random.uniform(-1, 1)
        

        info = {"image": self.current_image}
        
        return next_state, reward, done, info