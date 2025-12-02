"""
Example Usage Script
Demonstrates how to use the Azure data loader and train models programmatically
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from azure_data_loader import AzurePublicDatasetLoader
from train import ModelTrainer


def example_1_quick_test():
    """Example 1: Quick test with small dataset."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Quick Test (Small Dataset)")
    print("="*70)

    # Initialize loader
    loader = AzurePublicDatasetLoader()

    # Download single file
    print("\n1. Downloading data...")
    downloaded = loader.download_vm_traces(
        trace_types=['vm_cpu_readings'],
        file_numbers=[46],
        total_files=125
    )

    # Preprocess with 10% sampling
    print("\n2. Preprocessing...")
    df = loader.preprocess_vm_data(
        cpu_files=downloaded['vm_cpu_readings'],
        sample_fraction=0.1  # Use only 10% for quick test
    )

    # Prepare sequences for 1-hour prediction
    print("\n3. Creating sequences...")
    data = loader.prepare_sequences_for_prediction(
        df,
        lookback_hours=24,        # Use last 24 hours
        prediction_horizon_hours=1  # Predict 1 hour ahead
    )

    # Save processed data
    loader.save_processed_data(data, 'quick_test_data.pkl')

    # Train models
    print("\n4. Training models...")
    trainer = ModelTrainer(results_dir='results_quick_test')

    models, metrics = trainer.train_all_models(
        data,
        model_names=['simple_gru', 'bigru'],  # Train only 2 models for speed
        epochs=20,
        batch_size=32
    )

    print("\n✓ Example 1 completed!")
    print(f"  Best model: {min(metrics, key=lambda x: x['rmse'])['model_name']}")


def example_2_full_pipeline():
    """Example 2: Full pipeline with multiple files."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Full Pipeline (Multiple Files)")
    print("="*70)

    loader = AzurePublicDatasetLoader()

    # Download multiple files
    print("\n1. Downloading data...")
    downloaded = loader.download_vm_traces(
        trace_types=['vm_cpu_readings'],
        file_numbers=[46, 47],  # Multiple files
        total_files=125
    )

    # Preprocess with full data
    print("\n2. Preprocessing...")
    df = loader.preprocess_vm_data(
        cpu_files=downloaded['vm_cpu_readings'],
        sample_fraction=1.0  # Use full data
    )

    # Prepare sequences
    print("\n3. Creating sequences...")
    data = loader.prepare_sequences_for_prediction(
        df,
        lookback_hours=24,
        prediction_horizon_hours=1
    )

    # Save processed data
    loader.save_processed_data(data, 'full_pipeline_data.pkl')

    # Train all models
    print("\n4. Training models...")
    trainer = ModelTrainer(results_dir='results_full_pipeline')

    models, metrics = trainer.train_all_models(
        data,
        model_names=['simple_gru', 'bigru', 'gru_cnn', 'cnn_gru_attention'],
        epochs=50,
        batch_size=32
    )

    print("\n✓ Example 2 completed!")
    print(f"  Best model: {min(metrics, key=lambda x: x['rmse'])['model_name']}")


def example_3_reuse_data():
    """Example 3: Reuse preprocessed data."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Reuse Preprocessed Data")
    print("="*70)

    loader = AzurePublicDatasetLoader()

    # Load previously saved data
    print("\n1. Loading preprocessed data...")
    try:
        data = loader.load_processed_data('quick_test_data.pkl')
    except FileNotFoundError:
        print("Preprocessed data not found. Run example 1 first.")
        return

    # Train different models on same data
    print("\n2. Training CNN-GRU-Attention model...")
    trainer = ModelTrainer(results_dir='results_cnn_gru_attention')

    models, metrics = trainer.train_all_models(
        data,
        model_names=['cnn_gru_attention'],  # Train only one model
        epochs=100,  # More epochs for better performance
        batch_size=64  # Larger batch size
    )

    print("\n✓ Example 3 completed!")
    print(f"  Model RMSE: {metrics[0]['rmse']:.4f}")


def example_4_custom_hyperparameters():
    """Example 4: Custom model hyperparameters."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Custom Hyperparameters")
    print("="*70)

    from models import build_simple_gru
    import tensorflow as tf

    loader = AzurePublicDatasetLoader()

    # Load data
    print("\n1. Loading preprocessed data...")
    try:
        data = loader.load_processed_data('quick_test_data.pkl')
    except FileNotFoundError:
        print("Preprocessed data not found. Run example 1 first.")
        return

    # Build custom model
    print("\n2. Building custom model...")
    input_shape = data['X_train'].shape[1:]

    custom_model = build_simple_gru(
        input_shape=input_shape,
        gru_units=256,      # Larger than default (128)
        dense_units=128,    # Larger than default (64)
        dropout_rate=0.3    # Higher dropout
    )

    print(f"  Model parameters: {custom_model.count_params():,}")

    # Train custom model
    print("\n3. Training custom model...")
    history = custom_model.fit(
        data['X_train'], data['y_train'],
        validation_data=(data['X_val'], data['y_val']),
        epochs=30,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        ],
        verbose=1
    )

    # Evaluate
    print("\n4. Evaluating...")
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import numpy as np

    y_pred = custom_model.predict(data['X_test'], verbose=0).flatten()
    rmse = np.sqrt(mean_squared_error(data['y_test'], y_pred))
    mae = mean_absolute_error(data['y_test'], y_pred)

    print(f"\n✓ Example 4 completed!")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")


def example_5_different_horizons():
    """Example 5: Train models for different prediction horizons."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Different Prediction Horizons")
    print("="*70)

    loader = AzurePublicDatasetLoader()

    # Download data
    print("\n1. Downloading data...")
    downloaded = loader.download_vm_traces(
        trace_types=['vm_cpu_readings'],
        file_numbers=[46]
    )

    # Preprocess
    print("\n2. Preprocessing...")
    df = loader.preprocess_vm_data(
        cpu_files=downloaded['vm_cpu_readings'],
        sample_fraction=0.1
    )

    # Train models for different horizons
    horizons = [1, 3, 6]  # 1, 3, and 6 hours ahead

    for horizon in horizons:
        print(f"\n3. Training for {horizon}-hour prediction...")

        # Prepare sequences
        data = loader.prepare_sequences_for_prediction(
            df,
            lookback_hours=24,
            prediction_horizon_hours=horizon
        )

        # Train
        trainer = ModelTrainer(results_dir=f'results_{horizon}h_ahead')

        models, metrics = trainer.train_all_models(
            data,
            model_names=['simple_gru'],
            epochs=20,
            batch_size=32
        )

        print(f"  {horizon}h ahead - RMSE: {metrics[0]['rmse']:.4f}")

    print("\n✓ Example 5 completed!")


def main():
    """Main function to run examples."""
    print("="*70)
    print("VM USAGE PREDICTION - EXAMPLE USAGE")
    print("="*70)

    print("\nAvailable examples:\n")
    print("1. Quick Test - Small dataset, fast training")
    print("2. Full Pipeline - Multiple files, all models")
    print("3. Reuse Data - Load preprocessed data and train")
    print("4. Custom Hyperparameters - Build custom model")
    print("5. Different Horizons - Train for 1h, 3h, 6h predictions")
    print("6. Run All Examples")
    print("7. Exit\n")

    choice = input("Enter choice (1-7): ").strip()

    examples = {
        '1': example_1_quick_test,
        '2': example_2_full_pipeline,
        '3': example_3_reuse_data,
        '4': example_4_custom_hyperparameters,
        '5': example_5_different_horizons
    }

    if choice == '6':
        for example_func in examples.values():
            try:
                example_func()
            except Exception as e:
                print(f"\n✗ Example failed: {e}")
                continue
    elif choice in examples:
        examples[choice]()
    elif choice == '7':
        print("Exiting.")
        return
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
