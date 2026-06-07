import os
import json
import pandas as pd
import tensorflow as tf

# Hardware Enforcement: set memory growth BEFORE anything else
gpus = tf.config.experimental.list_physical_devices('GPU')
if not gpus:
    raise RuntimeError("No GPUs found. ABORTING to prevent silent CPU fallback.")

try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("Memory growth successfully set for GPUs.")
except RuntimeError as e:
    print(e)

from src.config import DATA_DIR, ROOT_DIR
from src.data_preprocessing import create_dataset

def main():
    print("--- Phase 4: Training & Tuning ---")
    
    splits_dir = DATA_DIR / "splits"
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    
    with open(splits_dir / "class_weights.json", "r") as f:
        class_weights_str = json.load(f)
    class_weights = {int(k): v for k, v in class_weights_str.items()}
    print(f"Applying Class Weights: {class_weights}")
    
    train_ds = create_dataset(train_df, is_training=True)
    val_ds = create_dataset(val_df, is_training=False)
    
    models_dir = ROOT_DIR / "models"
    model_path = models_dir / "uncompiled_model.keras"
    model = tf.keras.models.load_model(model_path)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-6
    )
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        class_weight=class_weights,
        callbacks=[early_stopping, reduce_lr]
    )
    
    trained_model_path = models_dir / "trained_lesion_classifier.keras"
    model.save(trained_model_path)
    print(f"Trained model saved to {trained_model_path}")
    
    history_dict = history.history
    for k in history_dict:
        history_dict[k] = [float(val) for val in history_dict[k]]
        
    history_path = models_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=4)
    print(f"Training history saved to {history_path}")

    epoch1_loss = history_dict['loss'][0]
    epoch1_acc = history_dict['accuracy'][0]
    epoch1_val_loss = history_dict['val_loss'][0]
    epoch1_val_acc = history_dict['val_accuracy'][0]
    
    final_loss = history_dict['loss'][-1]
    final_acc = history_dict['accuracy'][-1]
    final_val_loss = history_dict['val_loss'][-1]
    final_val_acc = history_dict['val_accuracy'][-1]
    
    print("\n--- Training Summary ---")
    print(f"Epoch 1: loss={epoch1_loss:.4f}, acc={epoch1_acc:.4f}, val_loss={epoch1_val_loss:.4f}, val_acc={epoch1_val_acc:.4f}")
    print(f"Final Epoch: loss={final_loss:.4f}, acc={final_acc:.4f}, val_loss={final_val_loss:.4f}, val_acc={final_val_acc:.4f}")

if __name__ == "__main__":
    main()
