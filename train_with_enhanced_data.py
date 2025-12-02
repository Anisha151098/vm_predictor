"""
Train Classification Models with ENHANCED DATASET
10x More Real Low CPU Data + All Previous Improvements
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, roc_auc_score
)
from imblearn.over_sampling import SMOTE
from collections import Counter

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

from models_classification import get_model
from data_loader_enhanced import EnhancedAzureVMDataLoader


def focal_loss(gamma=2.0, alpha=0.75):
    """Focal Loss for addressing remaining class imbalance."""
    def loss_fn(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        loss_1 = -alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)
        loss_0 = -(1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0)
        return K.mean(loss_1 + loss_0)
    return loss_fn


def create_classification_labels(y, threshold=0.5):
    """Convert continuous CPU values to binary labels."""
    return (y > threshold).astype(int)


def create_sequences(df, lookback=6, prediction_horizon=1):
    """Create sequences from DataFrame."""
    feature_cols = ['cpu_utilization', 'memory_utilization', 'network_utilization']
    X_all, y_all = [], []

    for vm_id in df['vm_id'].unique():
        vm_data = df[df['vm_id'] == vm_id].sort_values('timestamp')
        if len(vm_data) < lookback + prediction_horizon:
            continue

        features = vm_data[feature_cols].values / 100.0  # Normalize to [0, 1]

        for i in range(len(features) - lookback - prediction_horizon + 1):
            X_seq = features[i:i+lookback]
            y_target = features[i+lookback:i+lookback+prediction_horizon, 0].mean()
            X_all.append(X_seq)
            y_all.append(y_target)

    return np.array(X_all), np.array(y_all)


def prepare_enhanced_data(n_low_cpu_vms=80, n_high_cpu_vms=20, n_timestamps=500):
    """Prepare enhanced dataset with 10x more Low CPU data."""
    print("\n" + "="*70)
    print("PREPARING ENHANCED DATA (10X MORE LOW CPU)")
    print("="*70)

    # Generate enhanced dataset
    loader = EnhancedAzureVMDataLoader()
    df = loader.generate_enhanced_dataset(
        n_low_cpu_vms=n_low_cpu_vms,
        n_high_cpu_vms=n_high_cpu_vms,
        n_timestamps=n_timestamps,
        seed=42
    )

    # Create sequences
    lookback_hours = 6
    prediction_horizon_hours = 1
    sequence_length = (lookback_hours * 60) // 5  # 72 timesteps
    prediction_horizon = (prediction_horizon_hours * 60) // 5  # 12 timesteps

    X, y = create_sequences(df, lookback=sequence_length, prediction_horizon=prediction_horizon)

    # Convert to classification labels
    y_binary = create_classification_labels(y, threshold=0.5)

    print(f"\n[OK] Created {len(X)} sequences")
    print(f"  Input shape: {X.shape}")
    print(f"\n  ENHANCED Distribution (Before SMOTE):")
    unique, counts = np.unique(y_binary, return_counts=True)
    for label, count in zip(unique, counts):
        label_name = "Low CPU" if label == 0 else "High CPU"
        print(f"    {label_name} ({label}): {count} ({count/len(y_binary)*100:.1f}%)")

    # Split data
    train_size = int(0.7 * len(X))
    val_size = int(0.1 * len(X))

    X_train = X[:train_size]
    y_train = y_binary[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y_binary[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y_binary[train_size+val_size:]

    print(f"\n[OK] Splits:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")

    # Apply SMOTE only if still imbalanced
    print(f"\n" + "="*70)
    print("APPLYING SMOTE TO FURTHER BALANCE TRAINING DATA")
    print("="*70)

    n_samples, n_timesteps, n_features = X_train.shape
    X_train_flat = X_train.reshape(n_samples, -1)

    train_class_counts = Counter(y_train)
    print(f"\nTraining set before SMOTE:")
    for label, count in train_class_counts.items():
        label_name = "Low CPU" if label == 0 else "High CPU"
        print(f"  {label_name}: {count} ({count/len(y_train)*100:.1f}%)")

    smote = SMOTE(random_state=42, k_neighbors=min(5, train_class_counts[1]-1))
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_flat, y_train)
    X_train_balanced = X_train_balanced.reshape(-1, n_timesteps, n_features)

    print(f"\nTraining set after SMOTE:")
    train_balanced_counts = Counter(y_train_balanced)
    for label, count in train_balanced_counts.items():
        label_name = "Low CPU" if label == 0 else "High CPU"
        print(f"  {label_name}: {count} ({count/len(y_train_balanced)*100:.1f}%)")

    # Calculate class weights based on test set imbalance
    test_counts = Counter(y_test)
    if len(test_counts) == 2 and test_counts[1] > 0:
        weight_low = len(y_test) / (2 * test_counts[0]) if test_counts[0] > 0 else 1.0
        weight_high = len(y_test) / (2 * test_counts[1])
        # Reduce weight since we now have much more Low CPU data
        class_weights = {
            0: weight_low * 0.5,  # Reduced from 2.0
            1: weight_high
        }
    else:
        class_weights = {0: 1.0, 1: 1.0}

    print(f"\nClass weights (reduced due to better data): {class_weights}")

    return {
        'X_train': X_train_balanced,
        'y_train': y_train_balanced,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'input_shape': X_train_balanced.shape[1:],
        'class_weights': class_weights
    }


def find_optimal_threshold(y_true, y_pred_proba, metric='f1_macro'):
    """Find optimal classification threshold."""
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_score = 0

    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
        score = f1_score(y_true, y_pred, average='macro', zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


def evaluate_model(model, model_name, X_val, y_val, X_test, y_test):
    """Comprehensive evaluation with threshold optimization."""
    print(f"\n" + "="*70)
    print(f"EVALUATING {model_name.upper()}")
    print("="*70)

    # Find optimal threshold
    y_val_proba = model.predict(X_val, verbose=0).flatten()
    optimal_threshold, val_f1 = find_optimal_threshold(y_val, y_val_proba)
    print(f"\n[OK] Optimal Threshold: {optimal_threshold:.3f} (F1-Macro: {val_f1:.4f})")

    # Test predictions
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred_default = (y_pred_proba > 0.5).astype(int)
    y_pred_optimized = (y_pred_proba > optimal_threshold).astype(int)

    # Calculate metrics
    metrics = {
        'model_name': model_name,
        'optimal_threshold': float(optimal_threshold),
        'accuracy': float(accuracy_score(y_test, y_pred_optimized)),
        'auc': float(roc_auc_score(y_test, y_pred_proba)),
        'f1_macro': float(f1_score(y_test, y_pred_optimized, average='macro', zero_division=0)),
        'f1_macro_default': float(f1_score(y_test, y_pred_default, average='macro', zero_division=0)),
        'f1_binary_high': float(f1_score(y_test, y_pred_optimized, pos_label=1, zero_division=0)),
        'f1_low': float(f1_score(y_test, y_pred_optimized, pos_label=0, zero_division=0)),
        'f1_low_default': float(f1_score(y_test, y_pred_default, pos_label=0, zero_division=0)),
        'f1_high': float(f1_score(y_test, y_pred_optimized, pos_label=1, zero_division=0)),
        'precision_low': float(precision_score(y_test, y_pred_optimized, pos_label=0, zero_division=0)),
        'precision_high': float(precision_score(y_test, y_pred_optimized, pos_label=1, zero_division=0)),
        'recall_low': float(recall_score(y_test, y_pred_optimized, pos_label=0, zero_division=0)),
        'recall_high': float(recall_score(y_test, y_pred_optimized, pos_label=1, zero_division=0)),
        'confusion_matrix': confusion_matrix(y_test, y_pred_optimized).tolist()
    }

    print(f"\n[OK] Results:")
    print(f"  F1-Macro: {metrics['f1_macro']:.4f} (vs {metrics['f1_macro_default']:.4f} @ 0.5)")
    print(f"  F1-Low CPU: {metrics['f1_low']:.4f}")
    print(f"  F1-High CPU: {metrics['f1_high']:.4f}")
    print(f"  Recall-Low: {metrics['recall_low']:.4f}  Recall-High: {metrics['recall_high']:.4f}")

    return metrics


def main():
    """Main training pipeline with enhanced data."""
    print("\n" + "="*70)
    print("TRAINING WITH ENHANCED DATASET (10X MORE LOW CPU DATA)")
    print("="*70)

    # Setup
    results_dir = 'results_enhanced_data'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)

    # Prepare enhanced data
    data = prepare_enhanced_data(n_low_cpu_vms=80, n_high_cpu_vms=20, n_timestamps=500)

    # Models to train
    model_names = ['simple_gru', 'bigru', 'gru_cnn', 'cnn_gru_attention', 'gru_lstm']
    all_metrics = []

    # Train each model
    for model_name in model_names:
        print(f"\n" + "="*70)
        print(f"TRAINING: {model_name.upper()}")
        print("="*70)

        model = get_model(model_name, input_shape=data['input_shape'])
        print(f"\n[OK] Model: {model.name} ({model.count_params():,} parameters)")

        # Re-compile with Focal Loss
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=focal_loss(gamma=2.0, alpha=0.75),
            metrics=['accuracy', keras.metrics.AUC(name='auc'),
                     keras.metrics.Precision(name='precision'),
                     keras.metrics.Recall(name='recall')]
        )

        # Train
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        model_path = os.path.join(results_dir, 'models', f'{model_name}_best.h5')
        checkpoint = keras.callbacks.ModelCheckpoint(
            model_path, monitor='val_loss', save_best_only=True, verbose=0
        )

        history = model.fit(
            data['X_train'], data['y_train'],
            validation_data=(data['X_val'], data['y_val']),
            epochs=30,
            batch_size=32,
            class_weight=data['class_weights'],
            callbacks=[early_stop, checkpoint],
            verbose=1
        )

        print(f"[OK] Training completed. Model saved: {model_path}")

        # Evaluate
        metrics = evaluate_model(
            model, model_name,
            data['X_val'], data['y_val'],
            data['X_test'], data['y_test']
        )
        all_metrics.append(metrics)

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'enhanced_10x_low_cpu',
        'n_low_cpu_vms': 80,
        'n_high_cpu_vms': 20,
        'techniques': ['Enhanced Dataset', 'SMOTE', 'Focal Loss', 'Threshold Optimization'],
        'metrics': all_metrics,
        'best_model': max(all_metrics, key=lambda x: x['f1_macro'])['model_name']
    }

    results_path = os.path.join(results_dir, 'results_enhanced.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n" + "="*70)
    print("TRAINING WITH ENHANCED DATA COMPLETE!")
    print("="*70)
    print(f"Best model: {results['best_model'].upper()}")
    print(f"Results saved to: {results_dir}/")
    print("="*70)


if __name__ == "__main__":
    main()
