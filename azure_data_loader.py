"""
Enhanced Azure Public Dataset Loader
Handles downloading and preprocessing real Azure VM traces for usage prediction
"""

import pandas as pd
import numpy as np
import requests
import gzip
import os
import io
from typing import Tuple, List, Dict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class AzurePublicDatasetLoader:
    """
    Load and preprocess Azure Public Dataset VM traces.
    Supports CPU, memory, and network utilization data.
    """

    def __init__(self, data_dir='azure_data'):
        """
        Initialize Azure data loader.

        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'raw'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'processed'), exist_ok=True)

        # Azure Public Dataset base URL (V1)
        self.base_url = 'https://azurepublicdatasettraces.blob.core.windows.net/azurepublicdataset/trace_data'

        self.scalers = {
            'cpu': MinMaxScaler(),
            'memory': MinMaxScaler(),
            'network': MinMaxScaler(),
            'usage': MinMaxScaler()
        }

    def download_file(self, url: str, filename: str, force_download: bool = False) -> str:
        """
        Download a single file from Azure blob storage.

        Args:
            url: URL to download from
            filename: Local filename to save to
            force_download: Re-download even if file exists

        Returns:
            Path to downloaded file
        """
        filepath = os.path.join(self.data_dir, 'raw', filename)

        if os.path.exists(filepath) and not force_download:
            print(f"[OK] File already exists: {filename}")
            return filepath

        print(f"Downloading {filename}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192

            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            print(f"[OK] Downloaded: {filename}")
            return filepath

        except Exception as e:
            print(f"[ERROR] Error downloading {filename}: {e}")
            return None

    def load_csv_gz(self, filepath: str, column_names: List[str] = None) -> pd.DataFrame:
        """
        Load a gzipped CSV file.

        Args:
            filepath: Path to .csv.gz file
            column_names: List of column names (if CSV has no header)

        Returns:
            DataFrame with loaded data
        """
        print(f"Loading {os.path.basename(filepath)}...")
        try:
            with gzip.open(filepath, 'rt') as f:
                if column_names:
                    df = pd.read_csv(f, header=None, names=column_names)
                else:
                    df = pd.read_csv(f)
            print(f"[OK] Loaded {len(df)} rows")
            return df
        except Exception as e:
            print(f"[ERROR] Error loading file: {e}")
            return None

    def download_vm_traces(
        self,
        trace_types: List[str] = ['vm_cpu_readings', 'vm_memory_readings'],
        file_numbers: List[int] = [46],
        total_files: int = 125,
        force_download: bool = False
    ) -> Dict[str, List[str]]:
        """
        Download VM trace files from Azure Public Dataset.

        Args:
            trace_types: Types of traces to download (cpu, memory)
            file_numbers: Which file numbers to download (0-124 for V1)
            total_files: Total number of files available
            force_download: Re-download existing files

        Returns:
            Dictionary mapping trace type to list of downloaded file paths
        """
        downloaded_files = {trace_type: [] for trace_type in trace_types}

        print(f"\n{'='*60}")
        print("Downloading Azure Public Dataset VM Traces")
        print(f"{'='*60}")

        for trace_type in trace_types:
            print(f"\nTrace Type: {trace_type}")
            for file_num in file_numbers:
                filename = f"{trace_type}-file-{file_num}-of-{total_files}.csv.gz"
                url = f"{self.base_url}/{trace_type}/{filename}"

                filepath = self.download_file(url, filename, force_download)
                if filepath:
                    downloaded_files[trace_type].append(filepath)

        return downloaded_files

    def preprocess_vm_data(
        self,
        cpu_files: List[str],
        memory_files: List[str] = None,
        sample_fraction: float = 1.0,
        time_window_minutes: int = 60
    ) -> pd.DataFrame:
        """
        Preprocess VM trace data and merge CPU, memory metrics.

        Args:
            cpu_files: List of CPU trace file paths
            memory_files: List of memory trace file paths
            sample_fraction: Fraction of data to use (for testing)
            time_window_minutes: Time window for aggregation

        Returns:
            Preprocessed DataFrame with merged metrics
        """
        print(f"\n{'='*60}")
        print("Preprocessing VM Trace Data")
        print(f"{'='*60}")

        # Load CPU data
        # Azure Public Dataset V1 vm_cpu_readings columns:
        # timestamp, vm_id, avg_cpu, min_cpu, max_cpu
        cpu_column_names = ['timestamp', 'vm_id', 'avg_cpu', 'min_cpu', 'max_cpu']

        print("\nLoading CPU readings...")
        cpu_dfs = []
        for filepath in cpu_files:
            df = self.load_csv_gz(filepath, column_names=cpu_column_names)
            if df is not None:
                # Sample by VM instead of by row to maintain VM data quality
                if sample_fraction < 1.0:
                    # Get unique VMs and sample them
                    unique_vms = df['vm_id'].unique()
                    sample_size = max(1, int(len(unique_vms) * sample_fraction))
                    sampled_vms = np.random.choice(unique_vms, size=sample_size, replace=False)
                    df = df[df['vm_id'].isin(sampled_vms)]
                    print(f"  Sampled {sample_size} VMs out of {len(unique_vms)}")
                cpu_dfs.append(df)

        if not cpu_dfs:
            raise ValueError("No CPU data loaded")

        cpu_data = pd.concat(cpu_dfs, ignore_index=True)
        print(f"[OK] Total CPU records: {len(cpu_data)}")

        # Azure Public Dataset V1 format:
        # Columns: timestamp, vm_id, avg_cpu, min_cpu, max_cpu
        # Or similar depending on version

        # Check columns
        print(f"  Columns: {list(cpu_data.columns)}")

        # Standardize column names
        cpu_data = self.standardize_column_names(cpu_data)

        # Load memory data if provided
        if memory_files:
            print("\nLoading memory readings...")
            memory_dfs = []
            for filepath in memory_files:
                df = self.load_csv_gz(filepath)
                if df is not None:
                    if sample_fraction < 1.0:
                        df = df.sample(frac=sample_fraction, random_state=42)
                    memory_dfs.append(df)

            if memory_dfs:
                memory_data = pd.concat(memory_dfs, ignore_index=True)
                print(f"[OK] Total memory records: {len(memory_data)}")
                memory_data = self.standardize_column_names(memory_data)

                # Merge CPU and memory data
                print("\nMerging CPU and memory data...")
                merged_data = pd.merge(
                    cpu_data,
                    memory_data,
                    on=['timestamp', 'vm_id'],
                    how='inner',
                    suffixes=('', '_mem')
                )
                print(f"[OK] Merged records: {len(merged_data)}")
            else:
                merged_data = cpu_data
        else:
            merged_data = cpu_data

        # Sort by VM and timestamp
        merged_data = merged_data.sort_values(['vm_id', 'timestamp'])

        # Calculate VM usage metric (combined utilization)
        merged_data = self.calculate_vm_usage(merged_data)

        # Filter VMs with sufficient data
        merged_data = self.filter_vms_by_data_quality(merged_data, min_records=350)

        print(f"\n[OK] Final preprocessed data shape: {merged_data.shape}")
        print(f"  Number of VMs: {merged_data['vm_id'].nunique()}")
        print(f"  Time range: {merged_data['timestamp'].min()} to {merged_data['timestamp'].max()}")

        return merged_data

    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names across different Azure dataset versions."""
        # Common Azure Public Dataset column names
        column_mapping = {
            'vmId': 'vm_id',
            'vm_Id': 'vm_id',
            'subscriptionId': 'subscription_id',
            'deploymentId': 'deployment_id',
            'avg_cpu': 'cpu_utilization',
            'avg_mem': 'memory_utilization',
            'avg_memory': 'memory_utilization',
            'assigned_memory': 'memory_utilization',
            'max_cpu': 'cpu_max',
            'min_cpu': 'cpu_min',
            'max_mem': 'memory_max',
            'min_mem': 'memory_min',
        }

        df = df.rename(columns=column_mapping)

        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')

        return df

    def calculate_vm_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate overall VM usage from available metrics.

        Args:
            df: DataFrame with VM metrics

        Returns:
            DataFrame with vm_usage column
        """
        # Initialize usage as 0
        df['vm_usage'] = 0.0
        weight_sum = 0.0

        # CPU utilization (most important)
        if 'cpu_utilization' in df.columns:
            df['vm_usage'] += 0.5 * df['cpu_utilization']
            weight_sum += 0.5

        # Memory utilization
        if 'memory_utilization' in df.columns:
            df['vm_usage'] += 0.3 * df['memory_utilization']
            weight_sum += 0.3

        # Network utilization (if available)
        if 'network_utilization' in df.columns:
            df['vm_usage'] += 0.2 * df['network_utilization']
            weight_sum += 0.2

        # Normalize if weights don't sum to 1
        if weight_sum > 0 and weight_sum != 1.0:
            df['vm_usage'] = df['vm_usage'] / weight_sum

        # If no valid metrics, use CPU as fallback
        if weight_sum == 0 and 'cpu_utilization' in df.columns:
            df['vm_usage'] = df['cpu_utilization']

        return df

    def filter_vms_by_data_quality(
        self,
        df: pd.DataFrame,
        min_records: int = 20,
        max_vms: int = None
    ) -> pd.DataFrame:
        """
        Filter VMs with sufficient data quality.

        Args:
            df: Input DataFrame
            min_records: Minimum number of records per VM
            max_vms: Maximum number of VMs to keep

        Returns:
            Filtered DataFrame
        """
        print(f"\nFiltering VMs (min records: {min_records})...")

        # Count records per VM
        vm_counts = df['vm_id'].value_counts()
        valid_vms = vm_counts[vm_counts >= min_records].index

        print(f"  VMs before filtering: {len(vm_counts)}")
        print(f"  VMs after filtering: {len(valid_vms)}")

        # Filter to valid VMs
        df_filtered = df[df['vm_id'].isin(valid_vms)].copy()

        # Limit number of VMs if specified
        if max_vms and len(valid_vms) > max_vms:
            selected_vms = valid_vms[:max_vms]
            df_filtered = df_filtered[df_filtered['vm_id'].isin(selected_vms)]
            print(f"  Limited to top {max_vms} VMs")

        return df_filtered

    def prepare_sequences_for_prediction(
        self,
        df: pd.DataFrame,
        lookback_hours: int = 24,
        prediction_horizon_hours: int = 1,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Dict:
        """
        Prepare sequences for VM usage prediction.

        Args:
            df: Preprocessed DataFrame
            lookback_hours: Hours of historical data to use
            prediction_horizon_hours: Hours ahead to predict
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation

        Returns:
            Dictionary with train/val/test splits
        """
        print(f"\n{'='*60}")
        print("Preparing Sequences for Training")
        print(f"{'='*60}")
        print(f"  Lookback: {lookback_hours} hours")
        print(f"  Prediction horizon: {prediction_horizon_hours} hour(s)")

        # Determine sampling frequency from data
        sample_df = df[df['vm_id'] == df['vm_id'].iloc[0]].copy()
        time_diffs = sample_df['timestamp'].diff().dropna()
        median_interval = time_diffs.median()

        print(f"  Median sampling interval: {median_interval}")

        # Calculate number of time steps
        steps_per_hour = int(pd.Timedelta(hours=1) / median_interval)
        sequence_length = lookback_hours * steps_per_hour
        prediction_horizon = prediction_horizon_hours * steps_per_hour

        print(f"  Sequence length: {sequence_length} time steps")
        print(f"  Prediction horizon: {prediction_horizon} time steps")

        # Feature columns
        feature_cols = []
        if 'cpu_utilization' in df.columns:
            feature_cols.append('cpu_utilization')
        if 'memory_utilization' in df.columns:
            feature_cols.append('memory_utilization')
        if 'network_utilization' in df.columns:
            feature_cols.append('network_utilization')

        print(f"  Features: {feature_cols}")

        X_all, y_all = [], []

        # Process each VM separately
        for vm_id in tqdm(df['vm_id'].unique(), desc="Processing VMs"):
            vm_data = df[df['vm_id'] == vm_id].copy()

            # Skip if insufficient data
            if len(vm_data) < sequence_length + prediction_horizon:
                continue

            # Extract features
            features = vm_data[feature_cols].values
            target = vm_data['vm_usage'].values

            # Normalize features per VM
            features_scaled = features.copy()
            for i, col in enumerate(feature_cols):
                scaler = MinMaxScaler()
                features_scaled[:, i] = scaler.fit_transform(features[:, i].reshape(-1, 1)).flatten()

            # Normalize target
            target_scaler = MinMaxScaler()
            target_scaled = target_scaler.fit_transform(target.reshape(-1, 1)).flatten()

            # Create sequences
            X_seq, y_seq = self.create_sequences(
                features_scaled,
                target_scaled,
                sequence_length,
                prediction_horizon
            )

            X_all.append(X_seq)
            y_all.append(y_seq)

        # Combine all VMs
        X = np.vstack(X_all)
        y = np.concatenate(y_all)

        print(f"\n[OK] Created {len(X)} sequences")
        print(f"  Input shape: {X.shape}")
        print(f"  Target shape: {y.shape}")

        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=True, random_state=42
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), shuffle=True, random_state=42
        )

        print(f"\n[OK] Data splits:")
        print(f"  Training: {X_train.shape[0]} samples")
        print(f"  Validation: {X_val.shape[0]} samples")
        print(f"  Test: {X_test.shape[0]} samples")

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'feature_names': feature_cols,
            'sequence_length': sequence_length,
            'prediction_horizon': prediction_horizon,
            'scalers': self.scalers
        }

    def create_sequences(
        self,
        features: np.ndarray,
        target: np.ndarray,
        sequence_length: int,
        prediction_horizon: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.

        Args:
            features: Feature array (time_steps, n_features)
            target: Target array (time_steps,)
            sequence_length: Number of time steps to look back
            prediction_horizon: Number of time steps to predict ahead

        Returns:
            Tuple of (X, y) sequences
        """
        X, y = [], []

        for i in range(len(features) - sequence_length - prediction_horizon + 1):
            X.append(features[i:i + sequence_length])
            y.append(target[i + sequence_length + prediction_horizon - 1])

        return np.array(X), np.array(y)

    def save_processed_data(self, data_dict: Dict, filename: str = 'processed_azure_data.pkl'):
        """Save processed data to disk."""
        filepath = os.path.join(self.data_dir, 'processed', filename)
        with open(filepath, 'wb') as f:
            pickle.dump(data_dict, f)
        print(f"\n[OK] Saved processed data to: {filepath}")

    def load_processed_data(self, filename: str = 'processed_azure_data.pkl') -> Dict:
        """Load processed data from disk."""
        filepath = os.path.join(self.data_dir, 'processed', filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Processed data not found: {filepath}")

        with open(filepath, 'rb') as f:
            data_dict = pickle.load(f)
        print(f"[OK] Loaded processed data from: {filepath}")
        return data_dict


def main():
    """Example usage of Azure Public Dataset Loader."""
    print("Azure Public Dataset Loader - Example Usage")
    print("="*60)

    # Initialize loader
    loader = AzurePublicDatasetLoader()

    # Download sample files (start with just one file for testing)
    downloaded = loader.download_vm_traces(
        trace_types=['vm_cpu_readings'],
        file_numbers=[46],  # Start with one file
        total_files=125
    )

    # Preprocess data
    df = loader.preprocess_vm_data(
        cpu_files=downloaded['vm_cpu_readings'],
        sample_fraction=0.1  # Use 10% for initial testing
    )

    # Prepare sequences for training
    data = loader.prepare_sequences_for_prediction(
        df,
        lookback_hours=24,
        prediction_horizon_hours=1
    )

    # Save processed data
    loader.save_processed_data(data)

    print("\n[OK] Data loading and preprocessing completed!")
    print(f"  Ready for training with input shape: {data['X_train'].shape[1:]}")


if __name__ == "__main__":
    main()
