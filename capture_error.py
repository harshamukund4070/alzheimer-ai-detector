import os
import tensorflow as tf
import traceback

def capture_error():
    base_dir = os.getcwd()
    model_path = os.path.join(base_dir, "detector", "models", "alzheimer_model.h5")
    
    output_path = os.path.join(base_dir, "error_traceback.txt")
    with open(output_path, "w") as f:
        try:
            print("Attempting to load model...", file=f)
            model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully!", file=f)
        except Exception as e:
            print(f"Error: {e}", file=f)
            traceback.print_exc(file=f)
    print(f"Traceback saved to {output_path}")

if __name__ == "__main__":
    capture_error()
