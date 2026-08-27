import os
import tensorflow as tf

def load_model(filepath=None, default_path='raw_data/models/cnn_fracture_20260827-095048.keras'):
    """
    Loads a Keras model from a local path.
    """
    # 1. Check if the file/path exists locally
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No model found at path: '{filepath}'")

    print(f"Loading model from: {filepath} ...")

    try:
        # 2. Load the model
        model = tf.keras.models.load_model(filepath)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None
