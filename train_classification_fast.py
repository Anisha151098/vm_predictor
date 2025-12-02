"""
FAST Classification Training Script
Optimized version - completes in 3-4 hours for all 5 models

Key optimizations:
1. Smaller dataset (10 VMs instead of 100)
2. Reduced augmentation (2x instead of 4x)
3. SMOTE instead of ADASYN (faster)
4. Fewer epochs with early stopping
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
    confusion_matrix, roc_auc_score, classification_report
)
from imblearn.over_sampling import SMOTE
from collections import Counter
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers, Model

from data_loader_enhanced import EnhancedAzureVMDataLoader
from models_classification import build_simple_gru_classifier, build_bigru_classifier


# =============================================================================
# SIMPLIFIED DATA AUGMENTATION
# =============================================================================

def time_warp(x, sigma=0.2, knot=4):
    """Time warping augmentation."""
    orig_steps = np.arange(x.shape[0])
    warp_steps = np.linspace(0, x.shape[0]-1, knot+2)
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot+2,))
    random_warps = np.clip(random_warps, 0.5, 1.5)
    time_warp = np.interp(orig_steps, warp_steps, random_warps)

    warped_x = np.zeros_like(x)
    for i in range(x.shape[1]):
        warped_x[:, i] = x[:, i] * time_warp
    return warped_x


def magnitude_warp(x, sigma=0.2):
    """Magnitude warping augmentation."""
    warping_path = gaussian_filter1d(np.random.randn(x.shape[0]) * sigma, 5)
    warping_path = (1 + warping_path).reshape(-1, 1)
    return x * warping_path


def augment_sequence_simple(x):
    """Apply single random augmentation."""
    if np.random.rand() < 0.5:
        return time_warp(x)
    else:
        return magnitude_warp(x)


def simple_data_augmentation(X, y, target_class=0, augmentation_factor=2):
    """
    Simplified data augmentation for minority class.
    Only 2x augmentation for speed.
    """
    print(f"\n" + "="*70)
    print(f"SIMPLE DATA AUGMENTATION FOR CLASS {target_class}")
    print("="*70)

    minority_indices = np.where(y == target_class)[0]
    majority_indices = np.where(y != target_class)[0]

    print(f"\nBefore augmentation:")
    print(f"  Class 0 (Low CPU): {len(minority_indices)} samples")
    print(f"  Class 1 (High CPU): {len(majority_indices)} samples")

    # Augment minority class
    augmented_X = []
    augmented_y = []

    for idx in minority_indices:
        augmented_X.append(X[idx])  # Original
        augmented_y.append(target_class)
        # Add augmented version
        aug_sample = augment_sequence_simple(X[idx])
        augmented_X.append(aug_sample)
        augmented_y.append(target_class)

    # Combine with majority class
    X_combined = np.vstack([np.array(augmented_X), X[majority_indices]])
    y_combined = np.concatenate([np.array(augmented_y), y[majority_indices]])

    print(f"\nAfter augmentation:")
    print(f"  Class 0 (Low CPU): {np.sum(y_combined == 0)} samples")
    print(f"  Class 1 (High CPU): {np.sum(y_combined == 1)} samples")
    print(f"  Total: {len(X_combined)} samples")

    return X_combined, y_combined


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def focal_loss(gamma=2.0, alpha=0.75):
    """Focal Loss."""
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
# ADVANCED MODEL ARCHITECTURES
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


def get_advanced_model(input_shape, model_type='gru_attention_deep'):
    """Build advanced model architecture."""
    inputs = layers.Input(shape=input_shape)

    if model_type == 'gru_attention_deep':
        x = layers.GRU(128, return_sequences=True)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        x = layers.GRU(64, return_sequences=True)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        attended = attention_block(x)
        x = layers.GlobalAveragePooling1D()(attended)

        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(1, activation='sigmoid')(x)

    elif model_type == 'cnn_gru_attention_residual':
        cnn = layers.Conv1D(64, 3, padding='same', activation='relu')(inputs)
        cnn = layers.BatchNormalization()(cnn)
        cnn = layers.Conv1D(64, 3, padding='same', activation='relu')(cnn)
        cnn = layers.BatchNormalization()(cnn)
        cnn = layers.MaxPooling1D(2)(cnn)
        cnn = layers.Dropout(0.3)(cnn)

        gru = layers.GRU(64, return_sequences=True)(cnn)
        gru = layers.BatchNormalization()(gru)
        gru = layers.Dropout(0.3)(gru)

        if cnn.shape[1] != gru.shape[1]:
            cnn_residual = layers.Conv1D(64, 1, padding='same')(cnn)
        else:
            cnn_residual = cnn

        x = layers.Add()([gru, cnn_residual])
        attended = attention_block(x)
        x = layers.GlobalAveragePooling1D()(attended)

        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(1, activation='sigmoid')(x)

    elif model_type == 'ensemble_base':
        x = layers.GRU(64, return_sequences=True)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        attended = attention_block(x)
        x = layers.GlobalAveragePooling1D()(attended)

        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs, name=f'{model_type}_model')
    return model


# =============================================================================
# DATA PREPARATION
# =============================================================================

def create_sequences(df, lookback=6, prediction_horizon=1):
    """Create sequences from DataFrame."""
    feature_cols = ['cpu_utilization', 'memory_utilization', 'network_utilization']
    X_all, y_all = [], []

    for vm_id in df['vm_id'].unique():
        vm_data = df[df['vm_id'] == vm_id].sort_values('timestamp')
        if len(vm_data) < lookback + prediction_horizon:
            continue

        features = vm_data[feature_cols].values / 100.0

        for i in range(len(features) - lookback - prediction_horizon + 1):
            X_seq = features[i:i+lookback]
            y_target = features[i+lookback:i+lookback+prediction_horizon, 0].mean()
            X_all.append(X_seq)
            y_all.append(y_target)

    return np.array(X_all), np.array(y_all)


def create_classification_labels(y, threshold=0.5):
    """Convert continuous CPU values to binary labels."""
    return (y > threshold).astype(int)


def prepare_fast_data():
    """Prepare data with optimizations for speed."""
    print("\n" + "="*70)
    print("PREPARING FAST DATASET (OPTIMIZED)")
    print("="*70)

    # 1. Generate SMALLER enhanced dataset
    loader = EnhancedAzureVMDataLoader()
    df = loader.generate_enhanced_dataset(
        n_low_cpu_vms=8,    # REDUCED from 80
        n_high_cpu_vms=2,   # REDUCED from 20
        n_timestamps=500,
        seed=42
    )

    # 2. Create sequences
    lookback_hours = 6
    prediction_horizon_hours = 1
    sequence_length = (lookback_hours * 60) // 5
    prediction_horizon = (prediction_horizon_hours * 60) // 5

    X, y_continuous = create_sequences(df, lookback=sequence_length, prediction_horizon=prediction_horizon)
    y_binary = create_classification_labels(y_continuous, threshold=0.5)

    print(f"\n[OK] Created {len(X)} sequences")
    print(f"  Input shape: {X.shape}")

    # 3. Split data
    train_size = int(0.7 * len(X))
    val_size = int(0.1 * len(X))

    X_train = X[:train_size]
    y_train = y_binary[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y_binary[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y_binary[train_size+val_size:]

    print(f"\n[OK] Initial Splits:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")

    # 4. Simple Data Augmentation (2x only)
    X_train_aug, y_train_aug = simple_data_augmentation(
        X_train, y_train,
        target_class=0,
        augmentation_factor=2  # REDUCED from 4
    )

    # 5. Apply SMOTE (faster than ADASYN)
    print(f"\n" + "="*70)
    print("APPLYING SMOTE FOR FINAL BALANCING")
    print("="*70)

    n_samples, n_timesteps, n_features = X_train_aug.shape
    X_train_flat = X_train_aug.reshape(n_samples, -1)

    try:
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_flat, y_train_aug)
        X_train_balanced = X_train_balanced.reshape(-1, n_timesteps, n_features)
        print(f"\n[OK] SMOTE applied successfully")
    except Exception as e:
        print(f"\n[WARNING] SMOTE failed: {e}, using augmented data as-is")
        X_train_balanced = X_train_aug
        y_train_balanced = y_train_aug

    print(f"\nFinal Training Distribution:")
    train_counts = Counter(y_train_balanced)
    for label, count in train_counts.items():
        label_name = "Low CPU" if label == 0 else "High CPU"
        print(f"  {label_name}: {count} ({count/len(y_train_balanced)*100:.1f}%)")

    # 6. Class weights
    test_counts = Counter(y_test)
    if len(test_counts) == 2:
        class_weights = {0: 3.0, 1: 0.5}
    else:
        class_weights = {0: 1.0, 1: 1.0}

    print(f"\nClass weights: {class_weights}")

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


# =============================================================================
# EVALUATION
# =============================================================================

def find_optimal_threshold(y_true, y_pred_proba):
    """Find optimal threshold."""
    thresholds = np.arange(0.1, 0.9, 0.02)
    best_threshold = 0.5
    best_score = 0

    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_low = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
        recall_low = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        score = 0.4 * f1_macro + 0.4 * f1_low + 0.2 * recall_low

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


def evaluate_model(model, model_name, X_val, y_val, X_test, y_test):
    """Evaluate model."""
    print(f"\n" + "="*70)
    print(f"EVALUATING {model_name.upper()}")
    print("="*70)

    # Find optimal threshold
    y_val_proba = model.predict(X_val, verbose=0).flatten()
    optimal_threshold, val_score = find_optimal_threshold(y_val, y_val_proba)
    print(f"\n[OK] Optimal Threshold: {optimal_threshold:.3f} (Score: {val_score:.4f})")

    # Test predictions
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred_optimized = (y_pred_proba > optimal_threshold).astype(int)

    # Metrics
    metrics = {
        'model_name': model_name,
        'optimal_threshold': float(optimal_threshold),
        'accuracy': float(accuracy_score(y_test, y_pred_optimized)),
        'auc': float(roc_auc_score(y_test, y_pred_proba)),
        'f1_macro': float(f1_score(y_test, y_pred_optimized, average='macro', zero_division=0)),
        'f1_low': float(f1_score(y_test, y_pred_optimized, pos_label=0, zero_division=0)),
        'f1_high': float(f1_score(y_test, y_pred_optimized, pos_label=1, zero_division=0)),
        'precision_low': float(precision_score(y_test, y_pred_optimized, pos_label=0, zero_division=0)),
        'precision_high': float(precision_score(y_test, y_pred_optimized, pos_label=1, zero_division=0)),
        'recall_low': float(recall_score(y_test, y_pred_optimized, pos_label=0, zero_division=0)),
        'recall_high': float(recall_score(y_test, y_pred_optimized, pos_label=1, zero_division=0)),
        'confusion_matrix': confusion_matrix(y_test, y_pred_optimized).tolist()
    }

    print(f"\n[OK] Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  F1-Macro: {metrics['f1_macro']:.4f}")
    print(f"  F1-Low CPU: {metrics['f1_low']:.4f}")
    print(f"  F1-High CPU: {metrics['f1_high']:.4f}")
    print(f"  Recall-Low: {metrics['recall_low']:.4f}")
    print(f"  Recall-High: {metrics['recall_high']:.4f}")

    return metrics


# =============================================================================
# MAIN TRAINING
# =============================================================================

def main():
    """Main training pipeline - FAST version."""
    print("\n" + "="*70)
    print("FAST TRAINING: ALL 5 MODELS IN 3-4 HOURS")
    print("="*70)

    # Setup
    results_dir = 'results_fast'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'models'), exist_ok=True)

    # Prepare data
    data = prepare_fast_data()

    # Models to train
    model_configs = [
        {'name': 'simple_gru', 'epochs': 25, 'batch_size': 32, 'type': 'existing'},
        {'name': 'bigru', 'epochs': 25, 'batch_size': 32, 'type': 'existing'},
        {'name': 'gru_attention_deep', 'epochs': 30, 'batch_size': 32, 'type': 'advanced'},
        {'name': 'cnn_gru_attention_residual', 'epochs': 30, 'batch_size': 32, 'type': 'advanced'},
        {'name': 'ensemble_base', 'epochs': 25, 'batch_size': 32, 'type': 'advanced'}
    ]

    all_metrics = []

    for config in model_configs:
        model_name = config['name']
        model_type = config.get('type', 'advanced')

        print(f"\n" + "="*70)
        print(f"TRAINING: {model_name.upper()} ({model_type.upper()})")
        print("="*70)

        # Build model
        if model_type == 'existing':
            if model_name == 'simple_gru':
                model = build_simple_gru_classifier(data['input_shape'])
            elif model_name == 'bigru':
                model = build_bigru_classifier(data['input_shape'])
        else:
            model = get_advanced_model(data['input_shape'], model_type=model_name)

        print(f"\n[OK] Model: {model.name} ({model.count_params():,} parameters)")

        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=focal_loss(gamma=2.5, alpha=0.80),
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )

        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
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
            data['X_train'], data['y_train'],
            validation_data=(data['X_val'], data['y_val']),
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            class_weight=data['class_weights'],
            callbacks=[early_stop, reduce_lr, checkpoint],
            verbose=1
        )

        print(f"\n[OK] Training completed. Model saved: {model_path}")

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
        'dataset': 'fast_optimized',
        'total_training_samples': len(data['X_train']),
        'metrics': all_metrics
    }

    results_file = os.path.join(results_dir, 'results_fast.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)
    print(f"\n[OK] Results saved to: {results_file}")

    # Print summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    for m in all_metrics:
        print(f"\n{m['model_name'].upper()}:")
        print(f"  F1-Macro: {m['f1_macro']:.4f}")
        print(f"  F1-Low:   {m['f1_low']:.4f}  (Recall: {m['recall_low']:.4f})")
        print(f"  F1-High:  {m['f1_high']:.4f}  (Recall: {m['recall_high']:.4f})")


if __name__ == "__main__":
    main()
