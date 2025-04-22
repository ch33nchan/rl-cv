import numpy as np
from PIL import Image # Import Pillow
# Assuming tinygrad is accessible
from tinygrad.tensor import Tensor

class ImageProcessor:
    def __init__(self, target_size=(224, 224), normalize=True, interpolation=Image.Resampling.BILINEAR):
        self.target_size = target_size
        self.normalize = normalize
        self.interpolation = interpolation # Store interpolation method
        # Add other params like mean/std if normalizing

    def preprocess(self, image_np: np.ndarray) -> Tensor:
        """
        Preprocesses a single image (e.g., resize, normalize, convert to Tensor).
        Input: numpy array (H, W, C) in RGB format.
        Output: TinyGrad Tensor (1, C, H, W).
        """
        # Convert numpy array to PIL Image
        img = Image.fromarray(image_np.astype(np.uint8))

        # Resize using Pillow
        resized_img = img.resize(self.target_size, resample=self.interpolation)

        # Convert back to numpy array
        resized_image_np = np.array(resized_img)

        # Normalize (example)
        if self.normalize:
            processed_image = resized_image_np.astype(np.float32) / 255.0
            # Add mean/std normalization if needed (e.g., subtract mean, divide by std)
        else:
            processed_image = resized_image_np.astype(np.float32)

        # Transpose to (C, H, W)
        processed_image = np.transpose(processed_image, (2, 0, 1))

        # Convert to TinyGrad Tensor and add batch dimension
        # Ensure requires_grad is False unless inputs need gradients
        image_tensor = Tensor(processed_image, requires_grad=False).unsqueeze(0)
        return image_tensor

    # Add batch processing method if needed
    def preprocess_batch(self, images_np: list[np.ndarray]) -> Tensor:
         # Process each image individually and collect results
         processed_batch_list = []
         for img_np in images_np:
             # Convert numpy array to PIL Image
             img = Image.fromarray(img_np.astype(np.uint8))
             # Resize using Pillow
             resized_img = img.resize(self.target_size, resample=self.interpolation)
             # Convert back to numpy array
             resized_image_np = np.array(resized_img)

             # Normalize
             if self.normalize:
                 processed_image = resized_image_np.astype(np.float32) / 255.0
             else:
                 processed_image = resized_image_np.astype(np.float32)

             # Transpose to (C, H, W)
             processed_image = np.transpose(processed_image, (2, 0, 1))
             processed_batch_list.append(processed_image)

         # Stack along the batch dimension and convert to Tensor
         batch_array = np.stack(processed_batch_list, axis=0)
         batch_tensor = Tensor(batch_array, requires_grad=False)
         return batch_tensor