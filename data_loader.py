"""
Data Loader for Azure Public Dataset V1
Handles downloading, preprocessing, and preparing VM usage data for training
"""

import pandas as pd
import numpy as np
import requests
import os
from typing import Tuple, List
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle


class AzureVMDataLoader:
    """Load and preprocess Azure VM usage data."""

    def __init__(self, data_dir='data'):
        """
        Initialize data loader.

        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # Azure Public Dataset V1 VM trace links
        self.dataset_urls = {
            'vm_cpu_readings': 'https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/azuretracesv2.1/vm_cpu_readings-file-0-of-1.csv',
            'vm_memory_readings': 'https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/azuretracesv2.1/vm_memory_readings-file-0-of-1.csv',
        }

        self.scaler_cpu = MinMaxScaler()
        self.scaler_memory = MinMaxScaler()
        self.scaler_network = MinMaxScaler()

    def download_data(self, force_download=False):
        """
        Download Azure VM dataset.

        Args:
            force_download: If True, re-download even if files exist
        """
        print("Note: Azure Public Dataset is large (GBs). Using sample data for demonstration.")
        print("For production, download full dataset from:")
        print("https://github.com/Azure/AzurePublicDataset")

    def generate_sample_data(self, n_vms=10, n_timestamps=10000, seed=42):
        """
        Generate realistic sample VM usage data for demonstration.

        Args:
            n_vms: Number of VMs to simulate
            n_timestamps: Number of time steps
            seed: Random seed for reproducibility

        Returns:
            DataFrame with VM usage data
        """
        np.random.seed(seed)

        data = []

        for vm_id in range(n_vms):
            # Generate time series with realistic patterns
            timestamps = pd.date_range(start='2024-01-01', periods=n_timestamps, freq='5min')

            # CPU utilization: periodic pattern + noise
            t = np.arange(n_timestamps)
            cpu_base = 30 + 40 * np.sin(2 * np.pi * t / 288)  # Daily pattern
            cpu_util = np.clip(cpu_base + np.random.normal(0, 10, n_timestamps), 0, 100)

            # Memory utilization: slowly varying + noise
            memory_base = 50 + 20 * np.sin(2 * np.pi * t / 576)  # 2-day pattern
            memory_util = np.clip(memory_base + np.random.normal(0, 5, n_timestamps), 0, 100)

            # Network utilization: correlated with CPU + noise
            network_util = np.clip(cpu_util * 0.7 + np.random.normal(0, 15, n_timestamps), 0, 100)

            # VM usage (target): weighted combination
            vm_usage = (0.4 * cpu_util + 0.35 * memory_util + 0.25 * network_util)
            vm_usage = np.clip(vm_usage, 0, 100)

            for i in range(n_timestamps):
                data.append({
                    'vm_id': f'vm_{vm_id}',
                    'timestamp': timestamps[i],
                    'cpu_utilization': cpu_util[i],
                    'memory_utilization': memory_util[i],
                    'network_utilization': network_util[i],
                    'vm_usage': vm_usage[i]
                })

        df = pd.DataFrame(data)

        # Save to CSV
        filepath = os.path.join(self.data_dir, 'vm_usage_data.csv')
        df.to_csv(filepath, index=False)
        print(f"[OK] Generated sample data: {filepath}")
        print(f"  Shape: {df.shape}")
        print(f"  VMs: {n_vms}, Timestamps per VM: {n_timestamps}")

        return df

    def load_data(self, filepath=None):
        """
        Load VM usage data from CSV.

        Args:
            filepath: Path to CSV file. If None, uses default path.

        Returns:
            DataFrame with VM usage data
        """
        if filepath is None:
            filepath = os.path.join(self.data_dir, 'vm_usage_data.csv')

        if not os.path.exists(filepath):
            print("Data file not found. Generating sample data...")
            return self.generate_sample_data()

        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"[OK] Loaded data from: {filepath}")
        print(f"  Shape: {df.shape}")

        return df

    def create_sequences(
        self,
        data: np.ndarray,
        sequence_length: int,
        prediction_horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.

        Args:
            data: Input data array
            sequence_length: Number of time steps to look back
            prediction_horizon: Number of time steps to predict ahead

        Returns:
            Tuple of (X, y) sequences
        """
        X, y = [], []

        for i in range(len(data) - sequence_length - prediction_horizon + 1):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length + prediction_horizon - 1, -1])  # VM usage

        return np.array(X), np.array(y)

    def prepare_data(
        self,
        df: pd.DataFrame,
        sequence_length: int = 50,
        prediction_horizon: int = 1,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> dict:
        """
        Prepare data for model training.

        Args:
            df: DataFrame with VM usage data
            sequence_length: Number of time steps to look back
            prediction_horizon: Number of time steps to predict ahead
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation

        Returns:
            Dictionary with train/val/test splits and scalers
        """
        print(f"\nPreparing data for training...")
        print(f"  Sequence length: {sequence_length}")
        print(f"  Prediction horizon: {prediction_horizon}")

        # Select features and target
        features = ['cpu_utilization', 'memory_utilization', 'network_utilization', 'vm_usage']

        X_all, y_all = [], []

        # Process each VM separately to maintain temporal order
        for vm_id in df['vm_id'].unique():
            vm_data = df[df['vm_id'] == vm_id][features].values

            # Normalize
            vm_data_scaled = vm_data.copy()
            vm_data_scaled[:, 0] = self.scaler_cpu.fit_transform(vm_data[:, 0].reshape(-1, 1)).flatten()
            vm_data_scaled[:, 1] = self.scaler_memory.fit_transform(vm_data[:, 1].reshape(-1, 1)).flatten()
            vm_data_scaled[:, 2] = self.scaler_network.fit_transform(vm_data[:, 2].reshape(-1, 1)).flatten()
            vm_data_scaled[:, 3] = vm_data[:, 3]  # Keep target for now

            # Create sequences
            X_seq, y_seq = self.create_sequences(vm_data_scaled, sequence_length, prediction_horizon)

            X_all.append(X_seq)
            y_all.append(y_seq)

        X = np.vstack(X_all)
        y = np.concatenate(y_all)

        # Split into train, validation, and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), shuffle=False
        )

        print(f"[OK] Data prepared:")
        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Validation samples: {X_val.shape[0]}")
        print(f"  Test samples: {X_test.shape[0]}")
        print(f"  Input shape: {X_train.shape[1:]}")

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'feature_names': features[:-1],  # Exclude target
            'scalers': {
                'cpu': self.scaler_cpu,
                'memory': self.scaler_memory,
                'network': self.scaler_network
            }
        }

    def save_prepared_data(self, data_dict: dict, filepath='prepared_data.pkl'):
        """Save prepared data to disk."""
        filepath = os.path.join(self.data_dir, filepath)
        with open(filepath, 'wb') as f:
            pickle.dump(data_dict, f)
        print(f"[OK] Saved prepared data to: {filepath}")

    def load_prepared_data(self, filepath='prepared_data.pkl') -> dict:
        """Load prepared data from disk."""
        filepath = os.path.join(self.data_dir, filepath)
        with open(filepath, 'rb') as f:
            data_dict = pickle.load(f)
        print(f"[OK] Loaded prepared data from: {filepath}")
        return data_dict


if __name__ == "__main__":
    # Test data loader
    loader = AzureVMDataLoader()

    # Generate sample data
    df = loader.generate_sample_data(n_vms=10, n_timestamps=5000)

    # Prepare data
    data = loader.prepare_data(df, sequence_length=50)

    # Save prepared data
    loader.save_prepared_data(data)

    print("\n[OK] Data loader test completed successfully!")
