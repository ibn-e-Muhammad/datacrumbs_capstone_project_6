import tensorflow as tf
from src.config import IMG_HEIGHT, IMG_WIDTH, CHANNELS, ROOT_DIR
import os

def build_model():
    model = tf.keras.Sequential([
        # Input Layer implicitly defined via input_shape in the first layer or InputLayer
        tf.keras.layers.InputLayer(input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
        
        # Convolutional Block 1
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv2d_1'),
        tf.keras.layers.MaxPooling2D((2, 2), name='max_pooling2d_1'),
        
        # Convolutional Block 2
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2d_2'),
        tf.keras.layers.MaxPooling2D((2, 2), name='max_pooling2d_2'),
        
        # Convolutional Block 3
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv2d_3'),
        tf.keras.layers.MaxPooling2D((2, 2), name='max_pooling2d_3'),
        
        # Flattening
        tf.keras.layers.Flatten(name='flatten'),
        
        # Dense Block
        tf.keras.layers.Dense(128, activation='relu', name='dense_1'),
        tf.keras.layers.Dropout(0.5, name='dropout'),
        
        # Output Layer
        tf.keras.layers.Dense(1, activation='sigmoid', name='output')
    ], name="SkinLesion_CNN")
    
    return model

if __name__ == "__main__":
    print("--- Phase 3: CNN Architecture ---")
    model = build_model()
    model.summary()
    
    models_dir = ROOT_DIR / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Save the uncompiled model architecture
    model_path = models_dir / "uncompiled_model.keras"
    model.save(model_path)
    print(f"Uncompiled model saved successfully to {model_path}")
