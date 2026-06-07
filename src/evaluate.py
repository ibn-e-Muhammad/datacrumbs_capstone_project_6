import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay

# Disable GPU to force CPU on Windows if needed, since models were trained on Keras 2 Linux
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from src.config import ROOT_DIR, DATA_DIR
from src.data_preprocessing import create_dataset

def plot_learning_curves(history_path, results_dir):
    with open(history_path, 'r') as f:
        history = json.load(f)
        
    epochs = range(1, len(history['loss']) + 1)
    
    # Plot Loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history['loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(results_dir / 'loss_curve.png')
    plt.close()
    
    # Plot Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(results_dir / 'accuracy_curve.png')
    plt.close()
    print("Learning curves generated and saved to results/")

def main():
    print("--- Phase 5: Evaluation ---")
    models_dir = ROOT_DIR / "models"
    results_dir = ROOT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    
    # 1. Learning Curves
    history_path = models_dir / "training_history.json"
    plot_learning_curves(history_path, results_dir)
    
    # 2. Load Model (compile=False to bypass Keras 2 -> Keras 3 optimizer incompatibility)
    model_path = models_dir / "trained_lesion_classifier.keras"
    print("Loading model...")
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # 3. Load Test Data
    splits_dir = DATA_DIR / "splits"
    test_df = pd.read_csv(splits_dir / "test.csv")
    
    # Cross-OS path fix: The CSV contains Linux /mnt/d/ paths from Phase 4 manual override.
    if os.name == 'nt':
        test_df['image_path'] = test_df['image_path'].apply(lambda x: x.replace('/mnt/d/', 'd:/').replace('/', '\\'))
        
    print(f"Loaded test dataset: {len(test_df)} images.")
    test_ds = create_dataset(test_df, is_training=False)
    
    # 4. Inference
    print("Running inference on test dataset...")
    y_pred_probs = model.predict(test_ds)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    y_true = test_df['target'].values
    
    # 5. Metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print("\n--- Evaluation Metrics (Malignant Class) ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"TN: {tn} | FP: {fp}")
    print(f"FN: {fn} | TP: {tp}")
    
    # Plot Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign (0)', 'Malignant (1)'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Test Set')
    plt.savefig(results_dir / 'confusion_matrix.png')
    plt.close()
    print("Confusion matrix plot saved to results/confusion_matrix.png")
    
    # Save metrics to JSON for easy logging
    metrics = {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }
    with open(results_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()
