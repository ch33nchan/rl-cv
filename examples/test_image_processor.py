import numpy as np
import sys
import os

# Add the project root to the Python path to find modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

try:
    # Assuming tinygrad is installed or accessible via path
    from tinygrad.tensor import Tensor
    # Import the processor
    from modalities.image_processor import ImageProcessor
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure tinygrad is installed and the project structure is correct.")
    sys.exit(1)

def main():
    print("Testing ImageProcessor...")

    # 1. Create a dummy image (e.g., 480x640 RGB)
    dummy_height, dummy_width = 480, 640
    dummy_image_np = np.random.randint(0, 256, size=(dummy_height, dummy_width, 3), dtype=np.uint8)
    print(f"Created dummy image with shape: {dummy_image_np.shape} and dtype: {dummy_image_np.dtype}")

    # 2. Instantiate the ImageProcessor
    # Using default target size (224, 224) and normalization
    processor = ImageProcessor(target_size=(224, 224), normalize=True)
    print(f"Instantiated ImageProcessor with target_size={processor.target_size}, normalize={processor.normalize}")

    # 3. Preprocess the single dummy image
    try:
        processed_tensor = processor.preprocess(dummy_image_np)
        print("\n--- Single Image Preprocessing ---")
        print(f"Output tensor shape: {processed_tensor.shape}")
        print(f"Output tensor dtype: {processed_tensor.dtype}")
        # Optional: Print min/max values to check normalization
        # Need to convert back to numpy to easily get min/max if needed
        processed_np = processed_tensor.numpy()
        print(f"Output tensor min value: {np.min(processed_np):.4f}")
        print(f"Output tensor max value: {np.max(processed_np):.4f}")

    except Exception as e:
        print(f"Error during single image preprocessing: {e}")

    # 4. Preprocess a batch of dummy images
    try:
        batch_size = 4
        dummy_batch_np = [np.random.randint(0, 256, size=(dummy_height, dummy_width, 3), dtype=np.uint8) for _ in range(batch_size)]
        print(f"\n--- Batch Image Preprocessing (Batch Size: {batch_size}) ---")

        processed_batch_tensor = processor.preprocess_batch(dummy_batch_np)
        print(f"Output batch tensor shape: {processed_batch_tensor.shape}")
        print(f"Output batch tensor dtype: {processed_batch_tensor.dtype}")
        processed_batch_np = processed_batch_tensor.numpy()
        print(f"Output batch tensor min value: {np.min(processed_batch_np):.4f}")
        print(f"Output batch tensor max value: {np.max(processed_batch_np):.4f}")

    except Exception as e:
        print(f"Error during batch image preprocessing: {e}")


if __name__ == "__main__":
    main()