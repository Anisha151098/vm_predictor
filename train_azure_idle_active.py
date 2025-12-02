"""
Train GRU Models for IDLE vs ACTIVE VM Classification
Using Real Azure Public Dataset

Classification:
- IDLE: CPU utilization < 20%
- ACTIVE: CPU utilization >= 20%

Azure Data Format:
Column 0: timestamp
Column 1: vm_id (hashed)
Column 2: avg_cpu (%)
Column 3: max_cpu (%)
Column 4: min_cpu (%)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import gzip
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, roc_auc_score, classification_report
)
from collections import Counter

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers, Model

from models_classification import build_simple_gru_classifier, build_bigru_classifier


# =============================================================================
# DATA LOADING
# =============================================================================

def load_azure_data(filepath, max_vms=None, sample_frac=1.0):
    """
    Load Azure VM CPU data from gzipped CSV.

    Args:
        filepath: Path to .csv.gz file
        max_vms: Maximum number of VMs to load (None = all)
        sample_frac: Fraction of data to sample

    Returns:
        DataFrame with columns: timestamp, vm_id, avg_cpu, max_cpu, min_cpu
    """
    print(f"\n{'='*70}")
    print("LOADING AZURE PUBLIC DATASET")
    print(f"{'='*70}")
    print(f"File: {filepath}")

    column_names = ['timestamp', 'vm_id', 'avg_cpu', 'max_cpu', 'min_cpu']

    with gzip.open(filepath, 'rt') as f:
        df = pd.read_csv(f, header=None, names=column_names)

    print(f"\n[OK] Loaded {len(df):,} records")
    print(f"  Unique VMs: {df['vm_id'].nunique():,}")
    print(f"  Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Sample data if requested
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)
        print(f"\n[OK] Sampled to {len(df):,} records ({sample_frac*100:.1f}%)")

    # Limit number of VMs if requested
    if max_vms is not None:
        unique_vms = df['vm_id'].unique()[:max_vms]
        df = df[df['vm_id'].isin(unique_vms)]
        print(f"\n[OK] Limited to {max_vms} VMs, {len(df):,} records")

    # Sort by VM and timestamp
    df = df.sort_values(['vm_id', 'timestamp']).reset_index(drop=True)

    print(f"\n[OK] CPU Utilization Statistics:")
    print(f"  Mean: {df['avg_cpu'].mean():.2f}%")
    print(f"  Median: {df['avg_cpu'].median():.2f}%")
    print(f"  Std: {df['avg_cpu'].std():.2f}%")
    print(f"  Min: {df['avg_cpu'].min():.2f}%")
    print(f"  Max: {df['avg_cpu'].max():.2f}%")

    return df


def create_idle_active_labels(df, idle_threshold=20.0):
    """
    Create IDLE vs ACTIVE labels based on CPU threshold.

    Args:
        df: DataFrame with avg_cpu column
        idle_threshold: CPU % threshold (below = IDLE, above = ACTIVE)

    Returns:
        DataFrame with 'label' column (0 = IDLE, 1 = ACTIVE)
    """
    df['label'] = (df['avg_cpu'] >= idle_threshold).astype(int)

    idle_count = (df['label'] == 0).sum()
    active_count = (df['label'] == 1).sum()

    print(f"\n{'='*70}")
    print(f"IDLE vs ACTIVE CLASSIFICATION (Threshold: {idle_threshold}%)")
    print(f"{'='*70}")
    print(f"  IDLE (CPU < {idle_threshold}%): {idle_count:,} ({idle_count/len(df)*100:.1f}%)")
    print(f"  ACTIVE (CPU >= {idle_threshold}%): {active_count:,} ({active_count/len(df)*100:.1f}%)")

    return df


def create_sequences(df, lookback=12, prediction_horizon=1):
    """
    Create sequences for time series classification.

    Args:
        lookback: Number of past timesteps to use (12 = 1 hour at 5-min intervals)
        prediction_horizon: How far ahead to predict

    Returns:
        X: Input sequences (samples, timesteps, features)
        y: Labels (0 = IDLE, 1 = ACTIVE)
        vm_ids: VM IDs for each sequence
    """
    print(f"\n{'='*70}")
    print("CREATING SEQUENCES")
    print(f"{'='*70}")
    print(f"  Lookback: {lookback} timesteps ({lookback*5} minutes)")
    print(f"  Prediction horizon: {prediction_horizon} timestep(s) ({prediction_horizon*5} min)")

    X_all, y_all, vm_ids_all = [], [], []

    for vm_id in df['vm_id'].unique():
        vm_data = df[df['vm_id'] == vm_id].sort_values('timestamp')

        if len(vm_data) < lookback + prediction_horizon:
            continue

        # Features: avg_cpu, max_cpu, min_cpu (normalized to 0-1)
        features = vm_data[['avg_cpu', 'max_cpu', 'min_cpu']].values / 100.0
        labels = vm_data['label'].values

        for i in range(len(features) - lookback - prediction_horizon + 1):
            X_seq = features[i:i+lookback]
            # Predict label at future timestep
            y_label = labels[i+lookback+prediction_horizon-1]

            X_all.append(X_seq)
            y_all.append(y_label)
            vm_ids_all.append(vm_id)

    X = np.array(X_all)
    y = np.array(y_all)
    vm_ids = np.array(vm_ids_all)

    print(f"\n[OK] Created {len(X):,} sequences")
    print(f"  Input shape: {X.shape}")
    print(f"  Label distribution:")
    print(f"    IDLE: {np.sum(y == 0):,} ({np.sum(y == 0)/len(y)*100:.1f}%)")
    print(f"    ACTIVE: {np.sum(y == 1):,} ({np.sum(y == 1)/len(y)*100:.1f}%)")

    return X, y, vm_ids


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def focal_loss(gamma=2.0, alpha=0.75):
    """Focal Loss for handling class imbalance."""
    def loss_fn(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1 - epsilon)

        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        loss_1 = -alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)
        loss_0 = -(1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0)

        return K.mean(loss_1 + loss_0)
    return loss_fn


# =============================================================================
# ADVANCED MODELS
# =============================================================================

def attention_block(x):
    """Attention mechanism."""
    attention = layers.Dense(1, activation='tanh')(x)
    attention = layers.Flatten()(attention)
    attention = layers.Activation('softmax')(attention)
    attention = layers.RepeatVector(x.shape[-1])(attention)
    attention = layers.Permute([2, 1])(attention)
    attended = layers.Multiply()([x, attention])
    return attended


def build_gru_attention(input_shape):
    """GRU with Attention mechanism."""
    inputs = layers.Input(shape=input_shape)

    x = layers.GRU(64, return_sequences=True)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.GRU(32, return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Attention
    attended = attention_block(x)
    x = layers.GlobalAveragePooling1D()(attended)

    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs, name='gru_attention')
    return model


def build_cnn_gru(input_shape):
    """CNN-GRU hybrid model."""
    inputs = layers.Input(shape=input_shape)

    # CNN layers
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)

    # GRU layers
    x = layers.GRU(32, return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.GRU(16, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs, name='cnn_gru')
    return model


def build_cnn_gru_attention(input_shape):
    """CNN-GRU with Attention mechanism - Advanced hybrid model."""
    inputs = layers.Input(shape=input_shape)

    # CNN layers for local pattern extraction
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)

    # GRU layers for temporal dependencies
    x = layers.GRU(64, return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.GRU(32, return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Attention mechanism
    attended = attention_block(x)
    x = layers.GlobalAveragePooling1D()(attended)

    # Dense layers
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs, name='cnn_gru_attention')
    return model


# =============================================================================
# EVALUATION
# =============================================================================

def find_optimal_threshold(y_true, y_pred_proba):
    """Find optimal classification threshold."""
    thresholds = np.arange(0.1, 0.9, 0.02)
    best_threshold = 0.5
    best_f1 = 0

    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


def evaluate_model(model, model_name, X_val, y_val, X_test, y_test):
    """Evaluate model performance."""
    print(f"\n{'='*70}")
    print(f"EVALUATING {model_name.upper()}")
    print(f"{'='*70}")

    # Find optimal threshold on validation set
    y_val_proba = model.predict(X_val, verbose=0).flatten()
    optimal_threshold, val_f1 = find_optimal_threshold(y_val, y_val_proba)

    print(f"\n[OK] Optimal Threshold: {optimal_threshold:.3f} (Val F1: {val_f1:.4f})")

    # Test predictions
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba > optimal_threshold).astype(int)

    # Calculate metrics
    metrics = {
        'model_name': model_name,
        'optimal_threshold': float(optimal_threshold),
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'auc': float(roc_auc_score(y_test, y_pred_proba)),
        'f1_macro': float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
        'f1_idle': float(f1_score(y_test, y_pred, pos_label=0, zero_division=0)),
        'f1_active': float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        'precision_idle': float(precision_score(y_test, y_pred, pos_label=0, zero_division=0)),
        'precision_active': float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        'recall_idle': float(recall_score(y_test, y_pred, pos_label=0, zero_division=0)),
        'recall_active': float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

    # Print results
    print(f"\n[OK] Test Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  F1-Macro: {metrics['f1_macro']:.4f}")
    print(f"\n  IDLE Class:")
    print(f"    F1: {metrics['f1_idle']:.4f}")
    print(f"    Precision: {metrics['precision_idle']:.4f}")
    print(f"    Recall: {metrics['recall_idle']:.4f}")
    print(f"\n  ACTIVE Class:")
    print(f"    F1: {metrics['f1_active']:.4f}")
    print(f"    Precision: {metrics['precision_active']:.4f}")
    print(f"    Recall: {metrics['recall_active']:.4f}")

    cm = np.array(metrics['confusion_matrix'])
    print(f"\n  Confusion Matrix:")
    if cm.shape == (2, 2):
        print(f"    [[IDLE->IDLE={cm[0,0]:4d}  IDLE->ACTIVE={cm[0,1]:4d}]")
        print(f"     [ACTIVE->IDLE={cm[1,0]:4d}  ACTIVE->ACTIVE={cm[1,1]:4d}]]")
    else:
        print(f"    {cm}")
        print(f"    (Warning: Only one class present in test set)")

    return metrics


# =============================================================================
# MAIN TRAINING
# =============================================================================

def main():
    """Main training pipeline."""
    print("\n" + "="*70)
    print("AZURE VM IDLE/ACTIVE CLASSIFICATION")
    print("Using Real Azure Public Dataset")
    print("="*70)

    # Configuration
    AZURE_FILE = 'azure_data/raw/vm_cpu_readings-file-46-of-125.csv.gz'
    IDLE_THRESHOLD = 5.0  # CPU % threshold for IDLE classification (lowered to get better balance)
    MAX_VMS = 200  # More VMs for better representation
    SAMPLE_FRAC = 0.5  # Use 50% of data
    LOOKBACK = 12  # 1 hour of history (12 × 5min)
    EPOCHS = 25
    BATCH_SIZE = 64

    results_dir = 'results_azure_idle_active'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'models'), exist_ok=True)

    # 1. Load Azure data
    df = load_azure_data(AZURE_FILE, max_vms=MAX_VMS, sample_frac=SAMPLE_FRAC)

    # 2. Create IDLE/ACTIVE labels
    df = create_idle_active_labels(df, idle_threshold=IDLE_THRESHOLD)

    # 3. Create sequences
    X, y, vm_ids = create_sequences(df, lookback=LOOKBACK, prediction_horizon=1)

    # 4. Split data (70% train, 15% val, 15% test)
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]

    print(f"\n{'='*70}")
    print("DATA SPLITS")
    print(f"{'='*70}")
    print(f"  Training: {len(X_train):,} samples")
    print(f"    IDLE: {np.sum(y_train == 0):,} ({np.sum(y_train == 0)/len(y_train)*100:.1f}%)")
    print(f"    ACTIVE: {np.sum(y_train == 1):,} ({np.sum(y_train == 1)/len(y_train)*100:.1f}%)")
    print(f"  Validation: {len(X_val):,} samples")
    print(f"  Test: {len(X_test):,} samples")

    # 5. Calculate class weights
    class_counts = Counter(y_train)
    total = len(y_train)
    class_weights = {
        0: total / (2 * class_counts[0]),
        1: total / (2 * class_counts[1])
    }
    print(f"\n  Class weights: {class_weights}")

    # 6. Train models
    model_builders = {
        'simple_gru': lambda shape: build_simple_gru_classifier(shape),
        'bigru': lambda shape: build_bigru_classifier(shape),
        'gru_attention': lambda shape: build_gru_attention(shape),
        'cnn_gru': lambda shape: build_cnn_gru(shape),
        'cnn_gru_attention': lambda shape: build_cnn_gru_attention(shape)
    }

    all_metrics = []

    for model_name, build_fn in model_builders.items():
        print(f"\n{'='*70}")
        print(f"TRAINING: {model_name.upper()}")
        print(f"{'='*70}")

        # Build model
        model = build_fn(X_train.shape[1:])
        print(f"\n[OK] Model: {model.name} ({model.count_params():,} parameters)")

        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=focal_loss(gamma=2.0, alpha=0.7),
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )

        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )

        model_path = os.path.join(results_dir, 'models', f'{model_name}_best.h5')
        checkpoint = keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )

        # Train
        print(f"\nTraining {model_name}...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weights,
            callbacks=[early_stop, reduce_lr, checkpoint],
            verbose=1
        )

        print(f"\n[OK] Training completed. Model saved: {model_path}")

        # Evaluate
        metrics = evaluate_model(model, model_name, X_val, y_val, X_test, y_test)
        all_metrics.append(metrics)

    # 7. Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'azure_public_dataset_v1',
        'file': AZURE_FILE,
        'idle_threshold': IDLE_THRESHOLD,
        'max_vms': MAX_VMS,
        'sample_fraction': SAMPLE_FRAC,
        'lookback_timesteps': LOOKBACK,
        'total_sequences': len(X),
        'metrics': all_metrics
    }

    results_file = os.path.join(results_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print("TRAINING COMPLETED!")
    print(f"{'='*70}")
    print(f"\n[OK] Results saved to: {results_file}")

    # Print summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    for m in all_metrics:
        print(f"\n{m['model_name'].upper()}:")
        print(f"  F1-Macro: {m['f1_macro']:.4f}")
        print(f"  F1-IDLE:   {m['f1_idle']:.4f}  (Recall: {m['recall_idle']:.4f})")
        print(f"  F1-ACTIVE: {m['f1_active']:.4f}  (Recall: {m['recall_active']:.4f})")
        print(f"  AUC: {m['auc']:.4f}")


if __name__ == "__main__":
    main()
