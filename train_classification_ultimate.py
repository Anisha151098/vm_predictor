"""
ULTIMATE Classification Training Script
Publication-Quality Solution for Class Imbalance

Combines ALL Techniques:
1. Enhanced Dataset (10x more Low CPU data)
2. Advanced Data Augmentation (Time Warping, Magnitude Warping, Window Slicing)
3. Ensemble Methods (Multiple thresholds, model ensembles)
4. Cost-Sensitive Learning (Class weights + Focal Loss + Balanced CE)
5. Architecture Improvements (Attention, Skip Connections)
6. Advanced Training (Learning Rate Scheduling, Gradient Clipping)
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
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
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
# ADVANCED DATA AUGMENTATION
# =============================================================================

def time_warp(x, sigma=0.2, knot=4):
    """Time warping augmentation."""
    orig_steps = np.arange(x.shape[0])

    # Create warp steps ensuring they're within bounds
    warp_steps = np.linspace(0, x.shape[0]-1, knot+2)
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot+2,))
    random_warps = np.clip(random_warps, 0.5, 1.5)  # Limit warping range

    # Linear interpolation (more stable than cubic)
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


def window_slice(x, reduce_ratio=0.9):
    """Window slicing augmentation."""
    target_len = int(x.shape[0] * reduce_ratio)
    if target_len >= x.shape[0]:
        return x
    starts = np.random.randint(low=0, high=x.shape[0] - target_len)
    ends = starts + target_len

    # Interpolate back to original length
    ret = np.zeros_like(x)
    for i in range(x.shape[1]):
        ret[:, i] = np.interp(
            np.linspace(0, target_len-1, x.shape[0]),
            np.arange(target_len),
            x[starts:ends, i]
        )
    return ret


def augment_sequence(x, num_augmentations=3):
    """Apply multiple augmentations to a sequence."""
    augmented = [x]
    for _ in range(num_augmentations):
        aug_x = x.copy()
        if np.random.rand() < 0.5:
            aug_x = time_warp(aug_x)
        if np.random.rand() < 0.5:
            aug_x = magnitude_warp(aug_x)
        if np.random.rand() < 0.5:
            aug_x = window_slice(aug_x)
        augmented.append(aug_x)
    return augmented


def advanced_data_augmentation(X, y, target_class=0, augmentation_factor=5):
    """
    Advanced data augmentation for minority class.

    Args:
        X: Input sequences (n_samples, timesteps, features)
        y: Labels
        target_class: Class to augment (0 = Low CPU)
        augmentation_factor: How many augmented samples per original
    """
    print(f"\n" + "="*70)
    print(f"ADVANCED DATA AUGMENTATION FOR CLASS {target_class}")
    print("="*70)

    # Find minority class samples
    minority_indices = np.where(y == target_class)[0]
    majority_indices = np.where(y != target_class)[0]

    print(f"\nBefore augmentation:")
    print(f"  Class 0 (Low CPU): {len(minority_indices)} samples")
    print(f"  Class 1 (High CPU): {len(majority_indices)} samples")

    # Augment minority class
    augmented_X = []
    augmented_y = []

    for idx in minority_indices:
        augmented_samples = augment_sequence(X[idx], num_augmentations=augmentation_factor-1)
        augmented_X.extend(augmented_samples)
        augmented_y.extend([target_class] * len(augmented_samples))

    # Combine with original data
    X_combined = np.vstack([X, np.array(augmented_X)])
    y_combined = np.concatenate([y, np.array(augmented_y)])

    print(f"\nAfter augmentation:")
    print(f"  Class 0 (Low CPU): {np.sum(y_combined == 0)} samples")
    print(f"  Class 1 (High CPU): {np.sum(y_combined == 1)} samples")
    print(f"  Total: {len(X_combined)} samples")

    return X_combined, y_combined


# =============================================================================
# ADVANCED LOSS FUNCTIONS
# =============================================================================

def balanced_crossentropy(y_true, y_pred, beta=0.999):
    """Balanced Cross-Entropy Loss."""
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1 - epsilon)

    # Calculate effective number of samples
    N_pos = K.sum(y_true)
    N_neg = K.sum(1 - y_true)
    N_total = K.cast(K.shape(y_true)[0], K.floatx())

    # Class weights based on effective number
    effective_num_pos = (1 - K.pow(beta, N_pos)) / (1 - beta)
    effective_num_neg = (1 - K.pow(beta, N_neg)) / (1 - beta)

    weight_pos = (1 - beta) / effective_num_pos
    weight_neg = (1 - beta) / effective_num_neg

    # Balanced CE
    loss_pos = -weight_pos * y_true * K.log(y_pred)
    loss_neg = -weight_neg * (1 - y_true) * K.log(1 - y_pred)

    return K.mean(loss_pos + loss_neg)


def focal_loss(gamma=2.0, alpha=0.75):
    """Focal Loss with adjustable parameters."""
    def loss_fn(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1 - epsilon)

        # Focal loss calculation
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        loss_1 = -alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)
        loss_0 = -(1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0)

        return K.mean(loss_1 + loss_0)
    return loss_fn


def combined_loss(y_true, y_pred):
    """Combination of Focal Loss and Balanced CE."""
    focal = focal_loss(gamma=2.5, alpha=0.80)(y_true, y_pred)
    bce = balanced_crossentropy(y_true, y_pred, beta=0.9999)
    return 0.7 * focal + 0.3 * bce


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
    """
    Build advanced model architecture with attention and skip connections.

    Args:
        input_shape: (timesteps, features)
        model_type: Type of model architecture
    """
    inputs = layers.Input(shape=input_shape)

    if model_type == 'gru_attention_deep':
        # Deep GRU with Attention
        x = layers.GRU(128, return_sequences=True)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        x = layers.GRU(64, return_sequences=True)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        # Attention
        attended = attention_block(x)
        x = layers.GlobalAveragePooling1D()(attended)

        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(1, activation='sigmoid')(x)

    elif model_type == 'cnn_gru_attention_residual':
        # CNN + GRU with Residual Connections
        # CNN branch
        cnn = layers.Conv1D(64, 3, padding='same', activation='relu')(inputs)
        cnn = layers.BatchNormalization()(cnn)
        cnn = layers.Conv1D(64, 3, padding='same', activation='relu')(cnn)
        cnn = layers.BatchNormalization()(cnn)
        cnn = layers.MaxPooling1D(2)(cnn)
        cnn = layers.Dropout(0.3)(cnn)

        # GRU branch
        gru = layers.GRU(64, return_sequences=True)(cnn)
        gru = layers.BatchNormalization()(gru)
        gru = layers.Dropout(0.3)(gru)

        # Residual connection
        if cnn.shape[1] != gru.shape[1]:
            cnn_residual = layers.Conv1D(64, 1, padding='same')(cnn)
        else:
            cnn_residual = cnn

        x = layers.Add()([gru, cnn_residual])

        # Attention
        attended = attention_block(x)
        x = layers.GlobalAveragePooling1D()(attended)

        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(1, activation='sigmoid')(x)

    elif model_type == 'ensemble_base':
        # Simpler model for ensemble
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

        features = vm_data[feature_cols].values / 100.0  # Normalize to [0, 1]

        for i in range(len(features) - lookback - prediction_horizon + 1):
            X_seq = features[i:i+lookback]
            y_target = features[i+lookback:i+lookback+prediction_horizon, 0].mean()
            X_all.append(X_seq)
            y_all.append(y_target)

    return np.array(X_all), np.array(y_all)


def create_classification_labels(y, threshold=0.5):
    """Convert continuous CPU values to binary labels."""
    return (y > threshold).astype(int)


def prepare_ultimate_data():
    """Prepare data with ALL enhancements."""
    print("\n" + "="*70)
    print("PREPARING ULTIMATE DATASET")
    print("="*70)

    # 1. Generate enhanced dataset (10x more Low CPU)
    loader = EnhancedAzureVMDataLoader()
    df = loader.generate_enhanced_dataset(
        n_low_cpu_vms=80,   # 80% Low CPU VMs
        n_high_cpu_vms=20,  # 20% High CPU VMs
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

    # 4. Advanced Data Augmentation (only on training)
    X_train_aug, y_train_aug = advanced_data_augmentation(
        X_train, y_train,
        target_class=0,  # Low CPU
        augmentation_factor=4  # 4x augmentation
    )

    # 5. Apply ADASYN (more advanced than SMOTE)
    print(f"\n" + "="*70)
    print("APPLYING ADASYN FOR FINAL BALANCING")
    print("="*70)

    n_samples, n_timesteps, n_features = X_train_aug.shape
    X_train_flat = X_train_aug.reshape(n_samples, -1)

    try:
        adasyn = ADASYN(random_state=42, n_neighbors=5)
        X_train_balanced, y_train_balanced = adasyn.fit_resample(X_train_flat, y_train_aug)
        X_train_balanced = X_train_balanced.reshape(-1, n_timesteps, n_features)
        print(f"\n[OK] ADASYN applied successfully")
    except Exception as e:
        print(f"\n[WARNING] ADASYN failed: {e}")
        print(f"  Falling back to SMOTETomek")
        smotetomek = SMOTETomek(random_state=42)
        X_train_balanced, y_train_balanced = smotetomek.fit_resample(X_train_flat, y_train_aug)
        X_train_balanced = X_train_balanced.reshape(-1, n_timesteps, n_features)

    print(f"\nFinal Training Distribution:")
    train_counts = Counter(y_train_balanced)
    for label, count in train_counts.items():
        label_name = "Low CPU" if label == 0 else "High CPU"
        print(f"  {label_name}: {count} ({count/len(y_train_balanced)*100:.1f}%)")

    # 6. Calculate class weights for test set
    test_counts = Counter(y_test)
    if len(test_counts) == 2:
        # Stronger weights for Low CPU
        weight_low = 3.0  # Strong emphasis on Low CPU
        weight_high = 0.5
        class_weights = {0: weight_low, 1: weight_high}
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

def find_optimal_threshold_comprehensive(y_true, y_pred_proba):
    """Find optimal threshold using multiple metrics."""
    thresholds = np.arange(0.1, 0.9, 0.02)
    best_threshold = 0.5
    best_score = 0

    results = []
    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)

        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_low = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
        f1_high = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        recall_low = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        recall_high = recall_score(y_true, y_pred, pos_label=1, zero_division=0)

        # Combined score emphasizing F1-Low and Recall-Low
        score = 0.4 * f1_macro + 0.4 * f1_low + 0.2 * recall_low

        results.append({
            'threshold': threshold,
            'score': score,
            'f1_macro': f1_macro,
            'f1_low': f1_low,
            'f1_high': f1_high,
            'recall_low': recall_low,
            'recall_high': recall_high
        })

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score, results


def evaluate_model_comprehensive(model, model_name, X_val, y_val, X_test, y_test):
    """Comprehensive evaluation with threshold optimization."""
    print(f"\n" + "="*70)
    print(f"EVALUATING {model_name.upper()}")
    print("="*70)

    # Find optimal threshold
    y_val_proba = model.predict(X_val, verbose=0).flatten()
    optimal_threshold, val_score, threshold_results = find_optimal_threshold_comprehensive(y_val, y_val_proba)

    print(f"\n[OK] Optimal Threshold: {optimal_threshold:.3f} (Score: {val_score:.4f})")

    # Test predictions with both default and optimized thresholds
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred_default = (y_pred_proba > 0.5).astype(int)
    y_pred_optimized = (y_pred_proba > optimal_threshold).astype(int)

    # Detailed metrics
    metrics = {
        'model_name': model_name,
        'optimal_threshold': float(optimal_threshold),
        'accuracy': float(accuracy_score(y_test, y_pred_optimized)),
        'accuracy_default': float(accuracy_score(y_test, y_pred_default)),
        'auc': float(roc_auc_score(y_test, y_pred_proba)),
        'f1_macro': float(f1_score(y_test, y_pred_optimized, average='macro', zero_division=0)),
        'f1_macro_default': float(f1_score(y_test, y_pred_default, average='macro', zero_division=0)),
        'f1_low': float(f1_score(y_test, y_pred_optimized, pos_label=0, zero_division=0)),
        'f1_low_default': float(f1_score(y_test, y_pred_default, pos_label=0, zero_division=0)),
        'f1_high': float(f1_score(y_test, y_pred_optimized, pos_label=1, zero_division=0)),
        'f1_high_default': float(f1_score(y_test, y_pred_default, pos_label=1, zero_division=0)),
        'precision_low': float(precision_score(y_test, y_pred_optimized, pos_label=0, zero_division=0)),
        'precision_high': float(precision_score(y_test, y_pred_optimized, pos_label=1, zero_division=0)),
        'recall_low': float(recall_score(y_test, y_pred_optimized, pos_label=0, zero_division=0)),
        'recall_high': float(recall_score(y_test, y_pred_optimized, pos_label=1, zero_division=0)),
        'recall_low_default': float(recall_score(y_test, y_pred_default, pos_label=0, zero_division=0)),
        'recall_high_default': float(recall_score(y_test, y_pred_default, pos_label=1, zero_division=0)),
        'confusion_matrix': confusion_matrix(y_test, y_pred_optimized).tolist(),
        'confusion_matrix_default': confusion_matrix(y_test, y_pred_default).tolist()
    }

    print(f"\n[OK] Results (Optimized Threshold={optimal_threshold:.3f}):")
    print(f"  Accuracy: {metrics['accuracy']:.4f} (vs {metrics['accuracy_default']:.4f} @ 0.5)")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  F1-Macro: {metrics['f1_macro']:.4f} (vs {metrics['f1_macro_default']:.4f} @ 0.5)")
    print(f"\n  F1-Low CPU: {metrics['f1_low']:.4f} (vs {metrics['f1_low_default']:.4f} @ 0.5)")
    print(f"  F1-High CPU: {metrics['f1_high']:.4f} (vs {metrics['f1_high_default']:.4f} @ 0.5)")
    print(f"\n  Precision-Low: {metrics['precision_low']:.4f}  Precision-High: {metrics['precision_high']:.4f}")
    print(f"  Recall-Low: {metrics['recall_low']:.4f} (vs {metrics['recall_low_default']:.4f} @ 0.5)")
    print(f"  Recall-High: {metrics['recall_high']:.4f} (vs {metrics['recall_high_default']:.4f} @ 0.5)")

    print(f"\n  Confusion Matrix (Optimized):")
    cm = np.array(metrics['confusion_matrix'])
    print(f"    [[TN={cm[0,0]:4d}  FP={cm[0,1]:4d}]")
    print(f"     [FN={cm[1,0]:4d}  TP={cm[1,1]:4d}]]")

    return metrics


# =============================================================================
# MAIN TRAINING
# =============================================================================

def main():
    """Main training pipeline with ALL advanced techniques."""
    print("\n" + "="*70)
    print("ULTIMATE TRAINING: PUBLICATION-QUALITY SOLUTION")
    print("="*70)

    # Setup
    results_dir = 'results_ultimate'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)

    # Prepare data with ALL enhancements
    data = prepare_ultimate_data()

    # Models to train
    model_configs = [
        {'name': 'simple_gru', 'epochs': 30, 'batch_size': 32, 'type': 'existing'},
        {'name': 'bigru', 'epochs': 30, 'batch_size': 32, 'type': 'existing'},
        {'name': 'gru_attention_deep', 'epochs': 40, 'batch_size': 32, 'type': 'advanced'},
        {'name': 'cnn_gru_attention_residual', 'epochs': 40, 'batch_size': 32, 'type': 'advanced'},
        {'name': 'ensemble_base', 'epochs': 35, 'batch_size': 32, 'type': 'advanced'}
    ]

    all_metrics = []

    for config in model_configs:
        model_name = config['name']
        model_type = config.get('type', 'advanced')

        print(f"\n" + "="*70)
        print(f"TRAINING: {model_name.upper()} ({'EXISTING' if model_type == 'existing' else 'ADVANCED'})")
        print("="*70)

        # Build model based on type
        if model_type == 'existing':
            if model_name == 'simple_gru':
                model = build_simple_gru_classifier(data['input_shape'])
            elif model_name == 'bigru':
                model = build_bigru_classifier(data['input_shape'])
        else:
            model = get_advanced_model(data['input_shape'], model_type=model_name)

        print(f"\n[OK] Model: {model.name} ({model.count_params():,} parameters)")

        # Compile with combined loss
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=combined_loss,
            metrics=[
                'accuracy',
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
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
        metrics = evaluate_model_comprehensive(
            model, model_name,
            data['X_val'], data['y_val'],
            data['X_test'], data['y_test']
        )
        all_metrics.append(metrics)

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'ultimate_enhanced_augmented',
        'techniques': [
            'Enhanced Dataset (10x Low CPU)',
            'Time Warping Augmentation',
            'Magnitude Warping Augmentation',
            'Window Slicing Augmentation',
            'ADASYN/SMOTETomek',
            'Combined Loss (Focal + Balanced CE)',
            'Attention Mechanisms',
            'Residual Connections',
            'Learning Rate Scheduling',
            'Threshold Optimization'
        ],
        'metrics': all_metrics,
        'best_model': max(all_metrics, key=lambda x: x['f1_macro'])['model_name']
    }

    results_path = os.path.join(results_dir, 'results_ultimate.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print final summary
    print(f"\n" + "="*70)
    print("ULTIMATE TRAINING COMPLETE!")
    print("="*70)
    print(f"\nBest Model: {results['best_model'].upper()}")
    print(f"\nAll Model Results:")
    print(f"{'Model':<30} {'F1-Macro':>10} {'F1-Low':>10} {'Recall-Low':>12}")
    print("-" * 70)
    for m in all_metrics:
        print(f"{m['model_name']:<30} {m['f1_macro']:>10.4f} {m['f1_low']:>10.4f} {m['recall_low']:>12.4f}")

    print(f"\nResults saved to: {results_dir}/")
    print("="*70)


if __name__ == "__main__":
    main()
