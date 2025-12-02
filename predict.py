"""
Inference Script for VM Usage Prediction
Make predictions using trained models
"""

import os
import numpy as np
import argparse
from tensorflow import keras

from data_loader import AzureVMDataLoader


class VMUsagePredictor:
    """Make predictions using trained models."""

    def __init__(self, model_path: str):
        """
        Initialize predictor.

        Args:
            model_path: Path to trained model
        """
        self.model = keras.models.load_model(model_path, compile=False)
        self.loader = AzureVMDataLoader()

        print(f"✓ Loaded model from: {model_path}")

    def predict(
        self,
        cpu_util: np.ndarray,
        memory_util: np.ndarray,
        network_util: np.ndarray
    ) -> np.ndarray:
        """
        Make predictions for VM usage.

        Args:
            cpu_util: CPU utilization sequence (sequence_length,)
            memory_util: Memory utilization sequence (sequence_length,)
            network_util: Network utilization sequence (sequence_length,)

        Returns:
            Predicted VM usage
        """
        # Normalize inputs
        cpu_scaled = self.loader.scaler_cpu.transform(cpu_util.reshape(-1, 1)).flatten()
        memory_scaled = self.loader.scaler_memory.transform(memory_util.reshape(-1, 1)).flatten()
        network_scaled = self.loader.scaler_network.transform(network_util.reshape(-1, 1)).flatten()

        # Create input sequence (add placeholder for VM usage)
        X = np.stack([cpu_scaled, memory_scaled, network_scaled, np.zeros_like(cpu_scaled)], axis=1)
        X = X.reshape(1, *X.shape)  # Add batch dimension

        # Predict
        prediction = self.model.predict(X, verbose=0)

        return prediction[0, 0]

    def predict_batch(
        self,
        cpu_util: np.ndarray,
        memory_util: np.ndarray,
        network_util: np.ndarray
    ) -> np.ndarray:
        """
        Make batch predictions.

        Args:
            cpu_util: CPU utilization sequences (batch_size, sequence_length)
            memory_util: Memory utilization sequences (batch_size, sequence_length)
            network_util: Network utilization sequences (batch_size, sequence_length)

        Returns:
            Predicted VM usage (batch_size,)
        """
        batch_size = cpu_util.shape[0]
        predictions = []

        for i in range(batch_size):
            pred = self.predict(cpu_util[i], memory_util[i], network_util[i])
            predictions.append(pred)

        return np.array(predictions)


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='Predict VM usage')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--cpu', type=float, nargs='+', help='CPU utilization values')
    parser.add_argument('--memory', type=float, nargs='+', help='Memory utilization values')
    parser.add_argument('--network', type=float, nargs='+', help='Network utilization values')

    args = parser.parse_args()

    # Initialize predictor
    predictor = VMUsagePredictor(args.model)

    # Prepare inputs
    cpu = np.array(args.cpu)
    memory = np.array(args.memory)
    network = np.array(args.network)

    # Validate inputs
    if not (len(cpu) == len(memory) == len(network)):
        print("Error: All inputs must have the same length")
        return

    # Make prediction
    prediction = predictor.predict(cpu, memory, network)

    print(f"\n{'='*60}")
    print("VM Usage Prediction")
    print(f"{'='*60}")
    print(f"Input sequence length: {len(cpu)}")
    print(f"CPU utilization: mean={cpu.mean():.2f}, std={cpu.std():.2f}")
    print(f"Memory utilization: mean={memory.mean():.2f}, std={memory.std():.2f}")
    print(f"Network utilization: mean={network.mean():.2f}, std={network.std():.2f}")
    print(f"\nPredicted VM Usage: {prediction:.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
