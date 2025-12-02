"""Quick training with reduced lookback for sparse Azure data"""
import sys
sys.path.insert(0, '.')

from azure_data_loader import AzurePublicDatasetLoader
from data_loader import AzureVMDataLoader
from train import ModelTrainer

# Initialize loader
loader = AzurePublicDatasetLoader()

# Download
print("\n=== Downloading Azure Data ===")
downloaded = loader.download_vm_traces(
    trace_types=['vm_cpu_readings'],
    file_numbers=[46]
)

# Preprocess with reduced minimum (for 6-hour lookback need ~100 records)
print("\n=== Preprocessing ===")
df = loader.preprocess_vm_data(
    cpu_files=downloaded['vm_cpu_readings'],
    sample_fraction=0.3  # 30% sample
)

# Manually filter for VMs with at least 100 records
from collections import Counter
vm_counts = Counter(df['vm_id'])
valid_vms = [vm for vm, count in vm_counts.items() if count >= 100]
print(f"VMs with 100+ records: {len(valid_vms)}")

if len(valid_vms) == 0:
    print("ERROR: No VMs with enough data. Using sample data instead...")
    sample_loader = AzureVMDataLoader()
    df = sample_loader.generate_sample_data(n_vms=10, n_timestamps=500)
else:
    # Keep only valid VMs, limit to 10
    df = df[df['vm_id'].isin(valid_vms[:10])]

# Prepare with 6-hour lookback instead of 24
print("\n=== Creating Sequences (6h lookback, 1h prediction) ===")
data = loader.prepare_sequences_for_prediction(
    df,
    lookback_hours=6,  # Reduced from 24
    prediction_horizon_hours=1
)

# Train
print("\n=== Training Models ===")
trainer = ModelTrainer(results_dir='results_azure_quick')
models, metrics = trainer.train_all_models(
    data,
    model_names=['simple_gru', 'bigru'],
    epochs=10,
    batch_size=32
)

print("\n=== DONE ===")
print(f"Results in: results_azure_quick/")
