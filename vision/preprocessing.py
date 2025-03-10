import numpy as np

class ImagePreprocessor:
    
    
    def __init__(self, target_size=(64, 64), normalize=True, grayscale=False):

        self.target_size = target_size
        self.normalize = normalize
        self.grayscale = grayscale
        
    def process(self, image):

        if image is None:
            channels = 1 if self.grayscale else 3
            return np.random.rand(channels, *self.target_size).astype(np.float32)
        
        return np.random.rand(3, *self.target_size).astype(np.float32)