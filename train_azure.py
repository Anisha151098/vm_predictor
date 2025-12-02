"""
Main Training Script for Azure Public Dataset
Train GRU-based models for 1-hour ahead VM usage prediction
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import json
from datetime import datetime

from azure_data_loader import AzurePublicDatasetLoader
from train import ModelTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train VM usage prediction models on Azure Public Dataset'
    )

    # Data arguments
    parser.add_argument(
        '--file-numbers',
        type=int,
        nargs='+',
        default=[46],
        help='File numbers to download (e.g., 46 47 48)'
    )
    parser.add_argument(
        '--sample-fraction',
        type=float,
        default=1.0,
        help='Fraction of data to use (0.0-1.0), useful for testing'
    )
    parser.add_argument(
        '--max-vms',
        type=int,
        default=None,
        help='Maximum number of VMs to use for training'
    )
    parser.add_argument(
        '--force-download',
        action='store_true',
        help='Force re-download of data files'
    )

    # Sequence parameters
    parser.add_argument(
        '--lookback-hours',
        type=int,
        default=24,
        help='Hours of historical data to use for prediction'
    )
    parser.add_argument(
        '--prediction-horizon',
        type=int,
        default=1,
        help='Hours ahead to predict (default: 1 hour)'
    )

    # Training arguments
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['simple_gru', 'bigru', 'gru_cnn', 'cnn_gru_attention'],
        choices=['simple_gru', 'bigru', 'gru_cnn', 'cnn_gru_attention'],
        help='Models to train'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
    )

    # Output arguments
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results_azure',
        help='Directory to save results'
    )
    parser.add_argument(
        '--load-processed',
        type=str,
        default=None,
        help='Load preprocessed data from file (skips download/preprocessing)'
    )

    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_args()

    print("="*70)
    print("VM USAGE PREDICTION - AZURE PUBLIC DATASET")
    print("="*70)
    print(f"Configuration:")
    print(f"  File numbers: {args.file_numbers}")
    print(f"  Lookback: {args.lookback_hours} hours")
    print(f"  Prediction horizon: {args.prediction_horizon} hour(s)")
    print(f"  Models: {args.models}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sample fraction: {args.sample_fraction}")
    print("="*70)

    # Initialize data loader
    loader = AzurePublicDatasetLoader()

    # Load or prepare data
    if args.load_processed:
        print(f"\nLoading preprocessed data from {args.load_processed}...")
        data = loader.load_processed_data(args.load_processed)
    else:
        # Download data
        print("\n" + "="*70)
        print("STEP 1: Downloading Azure Public Dataset")
        print("="*70)

        downloaded = loader.download_vm_traces(
            trace_types=['vm_cpu_readings'],
            file_numbers=args.file_numbers,
            force_download=args.force_download
        )

        # Check if data was downloaded
        if not downloaded['vm_cpu_readings']:
            print("✗ No data downloaded. Exiting.")
            return

        # Preprocess data
        print("\n" + "="*70)
        print("STEP 2: Preprocessing Data")
        print("="*70)

        df = loader.preprocess_vm_data(
            cpu_files=downloaded['vm_cpu_readings'],
            sample_fraction=args.sample_fraction
        )

        # Filter to max VMs if specified
        if args.max_vms:
            unique_vms = df['vm_id'].unique()
            if len(unique_vms) > args.max_vms:
                selected_vms = unique_vms[:args.max_vms]
                df = df[df['vm_id'].isin(selected_vms)]
                print(f"Limited to {args.max_vms} VMs")

        # Prepare sequences
        print("\n" + "="*70)
        print("STEP 3: Creating Training Sequences")
        print("="*70)

        data = loader.prepare_sequences_for_prediction(
            df,
            lookback_hours=args.lookback_hours,
            prediction_horizon_hours=args.prediction_horizon
        )

        # Save processed data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_filename = f"processed_azure_data_{timestamp}.pkl"
        loader.save_processed_data(data, save_filename)
        print(f"  You can reuse this data with: --load-processed {save_filename}")

    # Train models
    print("\n" + "="*70)
    print("STEP 4: Training Models")
    print("="*70)

    trainer = ModelTrainer(results_dir=args.results_dir)

    models, metrics = trainer.train_all_models(
        data,
        model_names=args.models,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # Save configuration
    config = {
        'timestamp': datetime.now().isoformat(),
        'file_numbers': args.file_numbers,
        'lookback_hours': args.lookback_hours,
        'prediction_horizon_hours': args.prediction_horizon,
        'sequence_length': data['sequence_length'],
        'prediction_horizon_steps': data['prediction_horizon'],
        'sample_fraction': args.sample_fraction,
        'max_vms': args.max_vms,
        'models': args.models,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'input_shape': list(data['X_train'].shape[1:]),
        'training_samples': len(data['X_train']),
        'validation_samples': len(data['X_val']),
        'test_samples': len(data['X_test'])
    }

    config_path = os.path.join(args.results_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE")
    print("="*70)
    print(f"Results saved to: {args.results_dir}/")
    print(f"Configuration saved to: {config_path}")
    print(f"\nBest model: {min(metrics, key=lambda x: x['rmse'])['model_name']}")
    print("="*70)


if __name__ == "__main__":
    main()
