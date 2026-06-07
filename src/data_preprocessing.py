import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import json
from src.config import *

def load_and_map_metadata():
    # Load metadata
    df = pd.read_csv(METADATA_PATH)
    
    # Create the binary target column
    # Benign = 0 (NV, BKL, DF, VASC)
    # Malignant = 1 (MEL, BCC, AKIEC)
    df['is_malignant'] = df[['MEL', 'BCC', 'AKIEC']].sum(axis=1)
    df['target'] = (df['is_malignant'] > 0).astype(int)
    
    # Create absolute file paths
    df['image_path'] = df['image'].apply(lambda x: str(IMAGES_DIR / f"{x}.jpg"))
    
    # Verify existence
    df['exists'] = df['image_path'].apply(os.path.exists)
    missing = len(df) - df['exists'].sum()
    if missing > 0:
        print(f"Warning: {missing} images not found.")
        df = df[df['exists']]
        
    print(f"Total valid images: {len(df)}")
    return df

def split_data(df):
    # Train 70%, Val 15%, Test 15%
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['target'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['target'], random_state=42)
    
    print(f"Train split: {len(train_df)} | Val split: {len(val_df)} | Test split: {len(test_df)}")
    return train_df, val_df, test_df

def compute_class_weights(train_df):
    # Standard mathematical inverse-frequency formula
    # weight = Total / (Num_Classes * Class_Count)
    total = len(train_df)
    neg_count = len(train_df[train_df['target'] == 0])
    pos_count = len(train_df[train_df['target'] == 1])
    
    weight_for_0 = (1 / neg_count) * (total / 2.0)
    weight_for_1 = (1 / pos_count) * (total / 2.0)
    
    class_weights = {
        0: float(weight_for_0),
        1: float(weight_for_1)
    }
    
    print(f"Train Distribution -> Benign (0): {neg_count}, Malignant (1): {pos_count}")
    print(f"Computed Class Weights: {json.dumps(class_weights)}")
    return class_weights

def parse_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = img / 255.0
    return img, label

def augment_image(img, label):
    # Random flip, rotation approximation, and zoom approximation
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    return img, label

def create_dataset(df, is_training=False):
    paths = df['image_path'].values
    labels = df['target'].values
    
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size=1000)
        # Augmentation strictly applied ONLY to training set
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    else:
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.cache()
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
    return dataset

def main():
    print("--- Phase 2: Data Preprocessing ---")
    df = load_and_map_metadata()
    
    print(f"Overall Distribution -> Benign (0): {len(df[df['target'] == 0])}, Malignant (1): {len(df[df['target'] == 1])}")
    
    train_df, val_df, test_df = split_data(df)
    
    print("Computing class weights on training set ONLY...")
    class_weights = compute_class_weights(train_df)
    
    splits_dir = DATA_DIR / "splits"
    splits_dir.mkdir(exist_ok=True)
    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)
    
    with open(splits_dir / "class_weights.json", "w") as f:
        json.dump(class_weights, f)
        
    print("Data preprocessing complete and splits saved.")

if __name__ == "__main__":
    main()
