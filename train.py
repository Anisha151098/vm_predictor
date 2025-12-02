"""
Training Script for VM Usage Prediction Models
Train and evaluate multiple GRU-based models
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

from data_loader import AzureVMDataLoader
from models import get_model


class ModelTrainer:
    """Train and evaluate multiple models."""

    def __init__(self, results_dir='results'):
        """
        Initialize trainer.

        Args:
            results_dir: Directory to save results
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)

        self.results = {}

    def train_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 1
    ) -> tuple:
        """
        Train a single model.

        Args:
            model_name: Name of model to train
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level

        Returns:
            Tuple of (model, history)
        """
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()}")
        print(f"{'='*60}")

        # Build model
        input_shape = X_train.shape[1:]
        model = get_model(model_name, input_shape)

        # Print model summary
        if verbose > 0:
            model.summary()

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.results_dir, 'models', f'{model_name}_best.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]

        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        return model, history

    def evaluate_model(
        self,
        model,
        model_name: str,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> dict:
        """
        Evaluate model on test data.

        Args:
            model: Trained model
            model_name: Name of model
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\nEvaluating {model_name}...")

        # Make predictions
        y_pred = model.predict(X_test, verbose=0).flatten()

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100

        metrics = {
            'model_name': model_name,
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape)
        }

        print(f"[OK] {model_name} Evaluation:")
        print(f"  MSE:  {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")

        return metrics

    def plot_training_history(self, histories: dict):
        """
        Plot training history for all models.

        Args:
            histories: Dictionary of {model_name: history}
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        for model_name, history in histories.items():
            # Loss
            axes[0].plot(history.history['loss'], label=f'{model_name} - Train', alpha=0.7)
            axes[0].plot(history.history['val_loss'], label=f'{model_name} - Val', alpha=0.7, linestyle='--')

            # MAE
            axes[1].plot(history.history['mae'], label=f'{model_name} - Train', alpha=0.7)
            axes[1].plot(history.history['val_mae'], label=f'{model_name} - Val', alpha=0.7, linestyle='--')

        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Training and Validation MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'plots', 'training_history.png'), dpi=300, bbox_inches='tight')
        print(f"[OK] Saved training history plot")
        plt.close()

    def plot_predictions(self, models: dict, X_test: np.ndarray, y_test: np.ndarray, n_samples: int = 200):
        """
        Plot predictions vs actual values.

        Args:
            models: Dictionary of {model_name: model}
            X_test: Test features
            y_test: Test labels
            n_samples: Number of samples to plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()

        for idx, (model_name, model) in enumerate(models.items()):
            y_pred = model.predict(X_test[:n_samples], verbose=0).flatten()

            axes[idx].plot(y_test[:n_samples], label='Actual', alpha=0.7, linewidth=2)
            axes[idx].plot(y_pred, label='Predicted', alpha=0.7, linewidth=2)
            axes[idx].set_xlabel('Sample')
            axes[idx].set_ylabel('VM Usage')
            axes[idx].set_title(f'{model_name.upper()} - Predictions vs Actual')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'plots', 'predictions.png'), dpi=300, bbox_inches='tight')
        print(f"[OK] Saved predictions plot")
        plt.close()

    def plot_metrics_comparison(self, metrics_list: list):
        """
        Plot comparison of metrics across models.

        Args:
            metrics_list: List of metric dictionaries
        """
        df = pd.DataFrame(metrics_list)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        metrics = ['mse', 'rmse', 'mae', 'r2', 'mape']
        colors = sns.color_palette("husl", len(df))

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            bars = ax.bar(df['model_name'], df[metric], color=colors)
            ax.set_xlabel('Model')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'{metric.upper()} Comparison')
            ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)

        # Remove extra subplot
        fig.delaxes(axes[5])

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'plots', 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"[OK] Saved metrics comparison plot")
        plt.close()

    def save_results(self, metrics_list: list, histories: dict):
        """
        Save training results to JSON.

        Args:
            metrics_list: List of metric dictionaries
            histories: Dictionary of training histories
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics_list,
            'best_model': min(metrics_list, key=lambda x: x['rmse'])['model_name']
        }

        filepath = os.path.join(self.results_dir, 'results.json')
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n[OK] Saved results to: {filepath}")
        print(f"[OK] Best model (lowest RMSE): {results['best_model']}")

    def train_all_models(
        self,
        data: dict,
        model_names: list = None,
        epochs: int = 50,
        batch_size: int = 32
    ):
        """
        Train and evaluate all models.

        Args:
            data: Dictionary with train/val/test splits
            model_names: List of model names to train
            epochs: Number of training epochs
            batch_size: Batch size
        """
        if model_names is None:
            model_names = ['simple_gru', 'bigru', 'gru_cnn', 'cnn_gru_attention']

        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']
        X_test = data['X_test']
        y_test = data['y_test']

        trained_models = {}
        histories = {}
        metrics_list = []

        # Train each model
        for model_name in model_names:
            model, history = self.train_model(
                model_name, X_train, y_train, X_val, y_val,
                epochs=epochs, batch_size=batch_size, verbose=1
            )

            trained_models[model_name] = model
            histories[model_name] = history

            # Evaluate
            metrics = self.evaluate_model(model, model_name, X_test, y_test)
            metrics_list.append(metrics)

        # Generate visualizations
        print(f"\n{'='*60}")
        print("Generating visualizations...")
        print(f"{'='*60}")

        self.plot_training_history(histories)
        self.plot_predictions(trained_models, X_test, y_test)
        self.plot_metrics_comparison(metrics_list)

        # Save results
        self.save_results(metrics_list, histories)

        # Print summary
        print(f"\n{'='*60}")
        print("TRAINING SUMMARY")
        print(f"{'='*60}")

        df_metrics = pd.DataFrame(metrics_list)
        print(df_metrics.to_string(index=False))

        return trained_models, metrics_list


def main():
    """Main training function."""
    print("VM Usage Prediction - Model Training")
    print("="*60)

    # Load or generate data
    loader = AzureVMDataLoader()

    # Check if prepared data exists
    if os.path.exists('data/prepared_data.pkl'):
        print("Loading prepared data...")
        data = loader.load_prepared_data()
    else:
        print("Generating and preparing data...")
        df = loader.generate_sample_data(n_vms=10, n_timestamps=5000)
        data = loader.prepare_data(df, sequence_length=50)
        loader.save_prepared_data(data)

    # Initialize trainer
    trainer = ModelTrainer()

    # Train all models
    models, metrics = trainer.train_all_models(
        data,
        epochs=30,  # Reduced for faster training
        batch_size=32
    )

    print("\n[OK] Training completed successfully!")
    print(f"[OK] Results saved to: {trainer.results_dir}/")


if __name__ == "__main__":
    main()
