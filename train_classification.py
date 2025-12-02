"""
Train Classification Models for VM CPU Usage Prediction
Trains 5 models: Simple GRU, BiGRU, GRU+CNN, CNN-GRU-Attention, GRU+LSTM
Evaluates with comprehensive classification metrics
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

import tensorflow as tf
from tensorflow import keras

from models_classification import get_model
from data_loader import AzureVMDataLoader


def create_classification_labels(y, threshold=0.5):
    """
    Convert continuous CPU values to binary labels.
    High CPU (1) if above threshold, Low CPU (0) otherwise.
    """
    return (y > threshold).astype(int)


def prepare_classification_data(sample_size=10, n_timestamps=500):
    """Prepare data with classification labels."""
    print("\n" + "="*70)
    print("PREPARING CLASSIFICATION DATA")
    print("="*70)

    # Generate sample data
    loader = AzureVMDataLoader()
    df = loader.generate_sample_data(n_vms=sample_size, n_timestamps=n_timestamps, seed=42)
    print(f"[OK] Generated sample data: {len(df)} records")

    # Create sequences
    lookback_hours = 6
    prediction_horizon_hours = 1

    # Assuming 5-minute intervals
    sequence_length = (lookback_hours * 60) // 5  # 72 timesteps
    prediction_horizon = (prediction_horizon_hours * 60) // 5  # 12 timesteps

    feature_cols = ['cpu_utilization', 'memory_utilization', 'network_utilization']

    X_all, y_all = [], []

    for vm_id in df['vm_id'].unique():
        vm_data = df[df['vm_id'] == vm_id].sort_values('timestamp')

        if len(vm_data) < sequence_length + prediction_horizon:
            continue

        features = vm_data[feature_cols].values

        for i in range(len(features) - sequence_length - prediction_horizon + 1):
            X_seq = features[i:i+sequence_length]
            y_target = features[i+sequence_length:i+sequence_length+prediction_horizon, 0].mean()

            X_all.append(X_seq)
            y_all.append(y_target)

    X = np.array(X_all)
    y = np.array(y_all)

    # Convert to classification labels (High CPU: > 0.5, Low CPU: <= 0.5)
    y_binary = create_classification_labels(y, threshold=0.5)

    print(f"\n[OK] Created {len(X)} sequences")
    print(f"  Input shape: {X.shape}")
    print(f"  Target distribution:")
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

    print(f"\n[OK] Data splits:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'input_shape': X_train.shape[1:]
    }


def evaluate_classification_model(model, model_name, X_test, y_test):
    """Comprehensive classification evaluation."""
    print(f"\n" + "="*70)
    print(f"EVALUATING {model_name.upper()}")
    print("="*70)

    # Predictions
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_low = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    precision_high = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall_low = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    recall_high = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1_low = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1_high = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_binary = f1_score(y_test, y_pred, pos_label=1, zero_division=0)  # Binary F1 for High CPU

    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except:
        auc = 0.0

    cm = confusion_matrix(y_test, y_pred)

    # Print results
    print(f"\n[OK] Classification Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC: {auc:.4f}")
    print(f"\n  F1 Scores:")
    print(f"    F1-Macro: {f1_macro:.4f}")
    print(f"    F1-Binary (High CPU): {f1_binary:.4f}")
    print(f"    F1-Low CPU: {f1_low:.4f}")
    print(f"    F1-High CPU: {f1_high:.4f}")
    print(f"\n  Precision:")
    print(f"    Low CPU: {precision_low:.4f}")
    print(f"    High CPU: {precision_high:.4f}")
    print(f"\n  Recall:")
    print(f"    Low CPU: {recall_low:.4f}")
    print(f"    High CPU: {recall_high:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"    FN: {cm[1,0]}, TP: {cm[1,1]}")

    return {
        'model_name': model_name,
        'accuracy': float(accuracy),
        'auc': float(auc),
        'f1_macro': float(f1_macro),
        'f1_binary_high': float(f1_binary),
        'f1_low': float(f1_low),
        'f1_high': float(f1_high),
        'precision_low': float(precision_low),
        'precision_high': float(precision_high),
        'recall_low': float(recall_low),
        'recall_high': float(recall_high),
        'confusion_matrix': cm.tolist(),
        'y_true': y_test.tolist(),
        'y_pred': y_pred.tolist(),
        'y_pred_proba': y_pred_proba.tolist()
    }


def plot_confusion_matrices(all_metrics, results_dir):
    """Plot confusion matrices for all models."""
    print(f"\n" + "="*70)
    print("GENERATING CONFUSION MATRIX PLOTS")
    print("="*70)

    n_models = len(all_metrics)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, metrics in enumerate(all_metrics):
        cm = np.array(metrics['confusion_matrix'])

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Low CPU', 'High CPU'],
                   yticklabels=['Low CPU', 'High CPU'],
                   ax=axes[idx], cbar=True)

        axes[idx].set_title(f"{metrics['model_name']}\nAccuracy: {metrics['accuracy']:.3f}")
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')

    # Hide extra subplot
    if n_models < 6:
        axes[5].axis('off')

    plt.tight_layout()
    save_path = os.path.join(results_dir, 'plots', 'confusion_matrices.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {save_path}")


def plot_metrics_comparison(all_metrics, results_dir):
    """Plot comprehensive metrics comparison."""
    print(f"\n" + "="*70)
    print("GENERATING METRICS COMPARISON PLOTS")
    print("="*70)

    model_names = [m['model_name'] for m in all_metrics]

    # Create comprehensive comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. F1 Scores
    ax = axes[0, 0]
    f1_data = pd.DataFrame({
        'F1-Macro': [m['f1_macro'] for m in all_metrics],
        'F1-Binary (High)': [m['f1_binary_high'] for m in all_metrics],
        'F1-Low': [m['f1_low'] for m in all_metrics],
        'F1-High': [m['f1_high'] for m in all_metrics]
    }, index=model_names)
    f1_data.plot(kind='bar', ax=ax, rot=45, width=0.8)
    ax.set_title('F1 Scores Comparison', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)

    # 2. Precision
    ax = axes[0, 1]
    precision_data = pd.DataFrame({
        'Precision Low CPU': [m['precision_low'] for m in all_metrics],
        'Precision High CPU': [m['precision_high'] for m in all_metrics]
    }, index=model_names)
    precision_data.plot(kind='bar', ax=ax, rot=45, width=0.8)
    ax.set_title('Precision Comparison', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)

    # 3. Recall
    ax = axes[0, 2]
    recall_data = pd.DataFrame({
        'Recall Low CPU': [m['recall_low'] for m in all_metrics],
        'Recall High CPU': [m['recall_high'] for m in all_metrics]
    }, index=model_names)
    recall_data.plot(kind='bar', ax=ax, rot=45, width=0.8)
    ax.set_title('Recall Comparison', fontsize=12, fontweight='bold')
    ax.set_ylabel('Recall')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)

    # 4. Accuracy & AUC
    ax = axes[1, 0]
    acc_auc_data = pd.DataFrame({
        'Accuracy': [m['accuracy'] for m in all_metrics],
        'AUC': [m['auc'] for m in all_metrics]
    }, index=model_names)
    acc_auc_data.plot(kind='bar', ax=ax, rot=45, width=0.8)
    ax.set_title('Accuracy & AUC Comparison', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)

    # 5. Overall Performance Radar (F1-Macro)
    ax = axes[1, 1]
    x_pos = np.arange(len(model_names))
    f1_macros = [m['f1_macro'] for m in all_metrics]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(model_names)))
    bars = ax.bar(x_pos, f1_macros, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_title('Overall Performance (F1-Macro)', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Macro Score')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

    # 6. Best Model Summary
    ax = axes[1, 2]
    ax.axis('off')
    best_model = max(all_metrics, key=lambda x: x['f1_macro'])
    summary_text = f"""
BEST MODEL: {best_model['model_name']}

Key Metrics:
  Accuracy: {best_model['accuracy']:.4f}
  F1-Macro: {best_model['f1_macro']:.4f}
  AUC: {best_model['auc']:.4f}

  F1-Binary (High): {best_model['f1_binary_high']:.4f}
  F1-Low CPU: {best_model['f1_low']:.4f}
  F1-High CPU: {best_model['f1_high']:.4f}

  Precision (Low/High):
    {best_model['precision_low']:.4f} / {best_model['precision_high']:.4f}

  Recall (Low/High):
    {best_model['recall_low']:.4f} / {best_model['recall_high']:.4f}
"""
    ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round',
            facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    save_path = os.path.join(results_dir, 'plots', 'metrics_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {save_path}")


def main():
    """Main training pipeline."""
    print("\n" + "="*70)
    print("VM CPU CLASSIFICATION - ALL MODELS")
    print("="*70)

    # Setup
    results_dir = 'results_classification'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)

    # Prepare data
    data = prepare_classification_data(sample_size=10, n_timestamps=500)

    # Models to train
    model_names = ['simple_gru', 'bigru', 'gru_cnn', 'cnn_gru_attention', 'gru_lstm']

    all_metrics = []
    all_models = {}

    # Train each model
    for model_name in model_names:
        print(f"\n" + "="*70)
        print(f"TRAINING: {model_name.upper()}")
        print("="*70)

        # Build model
        model = get_model(model_name, input_shape=data['input_shape'])
        print(f"\n[OK] Model built: {model.name}")
        print(f"  Parameters: {model.count_params():,}")

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
            epochs=20,
            batch_size=32,
            callbacks=[early_stop, checkpoint],
            verbose=1
        )

        print(f"[OK] Training completed")
        print(f"[OK] Model saved: {model_path}")

        # Evaluate
        metrics = evaluate_classification_model(
            model, model_name, data['X_test'], data['y_test']
        )
        all_metrics.append(metrics)
        all_models[model_name] = model

    # Plot confusion matrices
    plot_confusion_matrices(all_metrics, results_dir)

    # Plot metrics comparison
    plot_metrics_comparison(all_metrics, results_dir)

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'models': model_names,
        'metrics': all_metrics,
        'best_model': max(all_metrics, key=lambda x: x['f1_macro'])['model_name']
    }

    results_path = os.path.join(results_dir, 'classification_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Results saved to: {results_dir}/")
    print(f"Best model: {results['best_model']}")
    print("="*70)


if __name__ == "__main__":
    main()
