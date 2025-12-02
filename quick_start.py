"""
Quick Start Script for Azure VM Usage Prediction
Provides easy-to-use commands for common training scenarios
"""

import subprocess
import sys
import os


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")


def run_command(cmd, description):
    """Run a command and handle errors."""
    print_header(description)
    print(f"Command: {' '.join(cmd)}\n")

    response = input("Run this command? (y/n): ").strip().lower()
    if response != 'y':
        print("Skipped.")
        return False

    try:
        result = subprocess.run(cmd, check=True)
        print("\n✓ Command completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Command failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        return False


def main():
    """Main quick start menu."""
    print_header("VM USAGE PREDICTION - QUICK START")

    print("Choose a training scenario:\n")
    print("1. Quick Test (5-10 min) - Small dataset, fast training")
    print("2. Development (30-60 min) - Medium dataset, good for testing")
    print("3. Production (2-4 hours) - Full dataset, best performance")
    print("4. Custom - Enter your own parameters")
    print("5. Reuse Preprocessed Data - Train with existing data")
    print("6. Exit\n")

    choice = input("Enter choice (1-6): ").strip()

    if choice == '1':
        # Quick test
        cmd = [
            sys.executable, 'train_azure.py',
            '--file-numbers', '46',
            '--sample-fraction', '0.1',
            '--max-vms', '10',
            '--epochs', '20',
            '--models', 'simple_gru', 'bigru'
        ]
        run_command(cmd, "QUICK TEST TRAINING")

    elif choice == '2':
        # Development
        cmd = [
            sys.executable, 'train_azure.py',
            '--file-numbers', '46', '47',
            '--sample-fraction', '0.5',
            '--epochs', '40',
            '--lookback-hours', '24',
            '--prediction-horizon', '1'
        ]
        run_command(cmd, "DEVELOPMENT TRAINING")

    elif choice == '3':
        # Production
        cmd = [
            sys.executable, 'train_azure.py',
            '--file-numbers', '46', '47', '48', '49', '50',
            '--sample-fraction', '1.0',
            '--epochs', '100',
            '--lookback-hours', '24',
            '--prediction-horizon', '1'
        ]
        run_command(cmd, "PRODUCTION TRAINING")

    elif choice == '4':
        # Custom
        print_header("CUSTOM TRAINING")

        file_nums = input("Enter file numbers (space-separated, e.g., 46 47 48): ").strip().split()
        sample_frac = input("Sample fraction (0.1 to 1.0): ").strip()
        epochs = input("Number of epochs (e.g., 50): ").strip()
        lookback = input("Lookback hours (e.g., 24): ").strip()
        horizon = input("Prediction horizon hours (e.g., 1): ").strip()

        print("\nAvailable models:")
        print("  1. simple_gru")
        print("  2. bigru")
        print("  3. gru_cnn")
        print("  4. cnn_gru_attention")
        models_choice = input("Enter model numbers (space-separated, e.g., 1 2 3 4): ").strip().split()

        model_map = {
            '1': 'simple_gru',
            '2': 'bigru',
            '3': 'gru_cnn',
            '4': 'cnn_gru_attention'
        }
        models = [model_map[m] for m in models_choice if m in model_map]

        cmd = [
            sys.executable, 'train_azure.py',
            '--file-numbers'] + file_nums + [
            '--sample-fraction', sample_frac,
            '--epochs', epochs,
            '--lookback-hours', lookback,
            '--prediction-horizon', horizon,
            '--models'] + models

        run_command(cmd, "CUSTOM TRAINING")

    elif choice == '5':
        # Reuse data
        print_header("REUSE PREPROCESSED DATA")

        # List available processed files
        data_dir = 'azure_data/processed'
        if os.path.exists(data_dir):
            files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
            if files:
                print("Available preprocessed files:\n")
                for i, f in enumerate(files, 1):
                    print(f"  {i}. {f}")
                file_choice = input("\nEnter file number: ").strip()
                try:
                    selected_file = files[int(file_choice) - 1]
                except (IndexError, ValueError):
                    print("Invalid choice.")
                    return
            else:
                print("No preprocessed files found.")
                return
        else:
            selected_file = input("Enter preprocessed data filename: ").strip()

        epochs = input("Number of epochs (e.g., 50): ").strip()

        print("\nAvailable models:")
        print("  1. simple_gru")
        print("  2. bigru")
        print("  3. gru_cnn")
        print("  4. cnn_gru_attention")
        models_choice = input("Enter model numbers (space-separated, e.g., 1 2 3 4): ").strip().split()

        model_map = {
            '1': 'simple_gru',
            '2': 'bigru',
            '3': 'gru_cnn',
            '4': 'cnn_gru_attention'
        }
        models = [model_map[m] for m in models_choice if m in model_map]

        cmd = [
            sys.executable, 'train_azure.py',
            '--load-processed', selected_file,
            '--epochs', epochs,
            '--models'] + models

        run_command(cmd, "TRAINING WITH PREPROCESSED DATA")

    elif choice == '6':
        print("Exiting.")
        return

    else:
        print("Invalid choice.")
        return

    # After training completes
    print_header("TRAINING COMPLETE")
    print("Results saved to: results_azure/")
    print("\nNext steps:")
    print("  1. Check results_azure/results.json for metrics")
    print("  2. View plots in results_azure/plots/")
    print("  3. Use saved models in results_azure/models/")
    print("\nRun this script again to train with different settings!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
