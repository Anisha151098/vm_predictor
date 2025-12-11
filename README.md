# VM State Prediction using GRU-Attention Architecture

**Thesis Project**: Advanced Deep Learning for Cloud Resource Optimization

Predict virtual machine (VM) states (IDLE vs ACTIVE) using GRU-Attention architecture with triple defense strategy for class imbalance. Achieves **93.65% accuracy** for 60-minute ahead forecasting  on real Azure datacenter workloads.

---

## Key Achievements

- **93.65% Forecasting Accuracy** - 60 minutes ahead VM state prediction
- **98.59% Recall on ACTIVE Class** - Critical for avoiding false consolidations
- **7× Parameter Efficiency** - 24,162 parameters (vs 168,609 for BiGRU)
- **Real Azure Data** - 156,031 VMs, 10M+ CPU readings
- **Triple Defense Strategy** - Focal Loss + Class Weighting + Threshold Optimization

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.11+
- TensorFlow 2.20.0
- NumPy 2.2.3
- Pandas 2.2.3
- Matplotlib 3.10.0
- scikit-learn 1.6.1

### 2. Prepare Azure Dataset

Download Azure Public Dataset V1 from [Azure Public Dataset](https://github.com/Azure/AzurePublicDataset) and place in `azure_data/` directory.

Expected structure:
```
azure_data/
├── vmtable.csv           # VM metadata
└── vm_cpu_readings-file-*.csv  # CPU utilization traces
```

### 3. Train Models

**For 60-minute ahead forecasting (main thesis task):**
```bash
python train_azure_forecast_1hr.py
```

```

**Expected training time:** 15-20 minutes on NVIDIA GPU, 45-60 minutes on CPU

### 4. View Results

Results are saved to:
- `results_azure_forecast_1hr/` - Forecasting results, models, metrics

Each directory contains:
- `results.json` - Performance metrics for all 5 models
- `*.keras` - Trained model files
- `*.png` - Confusion matrices, training curves
- `*_architecture.txt` - Model architecture details

---

## Architecture Overview

### GRU-Attention (Proposed Model)

```
Input(12, 3) → GRU(64) → BatchNorm → Dropout(0.3) →
GRU(32) → BatchNorm → Dropout(0.3) → Attention →
Dense(32, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)
```

**Parameters:** 24,162
**Key Innovation:** Bahdanau-style attention adapted for time series
**Advantage:** 7× fewer parameters than BiGRU with better accuracy

### Triple Defense Strategy

Handles severe class imbalance (81.6% IDLE / 18.4% ACTIVE):

1. **Focal Loss** - γ=2.0, α=0.7 (down-weights easy majority class)
2. **Class Weighting** - 0.61 (IDLE) / 2.87 (ACTIVE) during training
3. **Threshold Optimization** - 0.54-0.58 (optimized on validation set)

### All 5 Models

1. **Simple GRU** - Baseline (91.16% forecast accuracy)
2. **BiGRU** - Bidirectional (93.37% forecast accuracy)
3. **GRU-Attention** - Proposed (93.65% forecast accuracy) ⭐
4. **CNN-GRU** - Hybrid (92.54% forecast accuracy)
5. **CNN-GRU-Attention** - Complex (92.82% forecast accuracy)

---



## Usage Examples

### Load and Preprocess Data

```python
from azure_data_loader import AzureDataLoader

# Initialize loader
loader = AzureDataLoader(
    idle_threshold=5.0,        # CPU % threshold for IDLE
    lookback_timesteps=12,     # 60 minutes history
    prediction_horizon=12      # 60 minutes ahead
)

# Load and preprocess
loader.load_data()
loader.preprocess()
sequences = loader.create_sequences()

# Split into train/val/test
X_train, y_train, X_val, y_val, X_test, y_test = loader.split_data()
```

### Train GRU-Attention Model

```python
from models_classification import build_gru_attention_model

# Build model
model = build_gru_attention_model(
    input_shape=(12, 3),
    gru_units_1=64,
    gru_units_2=32,
    dense_units=32,
    dropout_rate=0.3
)

# Compile with Focal Loss
from train_azure_forecast_1hr import focal_loss

model.compile(
    optimizer='adam',
    loss=focal_loss(gamma=2.0, alpha=0.7),
    metrics=['accuracy']
)

# Train with class weights
class_weights = {0: 0.61, 1: 2.87}

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    class_weight=class_weights
)
```

### Generate Visualizations

```python
# Generate architecture diagrams
python create_architecture_diagrams.py

# Generate comparison graphs from results
python create_individual_graphs.py
```

---

## Results Summary

### Forecasting (60 minutes ahead)

| Model | Accuracy | F1-Macro | AUC | Recall (ACTIVE) | Parameters |
|-------|----------|----------|-----|-----------------|------------|
| Simple GRU | 91.16% | 84.98% | 95.06% | 97.18% | 39,841 |
| BiGRU | 93.37% | 88.69% | 96.12% | 97.18% | 168,609 |
| **GRU-Attention** | **93.65%** | **90.89%** | **96.59%** | **98.59%** | **24,162** |
| CNN-GRU | 92.54% | 87.32% | 95.68% | 97.18% | 52,065 |
| CNN-GRU-Attention | 92.82% | 87.84% | 95.97% | 97.18% | 34,369 |



## Documentation

**For complete details, see:**

1. **[CONFIGURATION_MANUAL.md](CONFIGURATION_MANUAL.md)** - Complete setup guide with troubleshooting
2. **[COMPLETE_METHODOLOGY.md](COMPLETE_METHODOLOGY.md)** - Full methodology suitable for academic publication
3. **[LITERATURE_REVIEW.md](LITERATURE_REVIEW.md)** - Critical analysis of 40+ related papers
4. **[SECTION_5_IMPLEMENTATION_FINAL.md](SECTION_5_IMPLEMENTATION_FINAL.md)** - Implementation chapter for thesis
5. **[VIDEO_PRESENTATION_SCRIPT.md](VIDEO_PRESENTATION_SCRIPT.md)** - Presentation script with code walkthrough
6. **[ARCHITECTURE_TEXT_DIAGRAMS.md](ARCHITECTURE_TEXT_DIAGRAMS.md)** - Text-based architecture diagrams
7. **[QUICKSTART.md](QUICKSTART.md)** - Quick reference guide
8. **[AZURE_TRAINING_GUIDE.md](AZURE_TRAINING_GUIDE.md)** - Azure-specific training instructions

---

## System Requirements

### Minimum Requirements
- **CPU:** 4 cores, 2.5 GHz
- **RAM:** 8 GB
- **Storage:** 10 GB free space
- **Python:** 3.11 or higher

### Recommended for Training
- **GPU:** NVIDIA GPU with 4+ GB VRAM (CUDA-compatible)
- **RAM:** 16 GB
- **Storage:** 20 GB SSD

### Expected Performance
- **Training Time (GPU):** 15-20 minutes per model
- **Training Time (CPU):** 45-60 minutes per model
- **Inference Time:** <10ms per sequence

---

## Key Features

- ✅ **Dual-Task Framework** - Classification + Forecasting
- ✅ **Triple Defense for Class Imbalance** - Focal Loss + Weighting + Threshold
- ✅ **5 State-of-the-Art Architectures** - GRU, BiGRU, Attention, CNN-hybrid
- ✅ **Real Azure Datacenter Data** - 156K VMs, 10M+ readings
- ✅ **Production-Ready** - Complete preprocessing, training, evaluation pipeline
- ✅ **Comprehensive Metrics** - Accuracy, F1, AUC, Precision, Recall, Confusion Matrix
- ✅ **Publication-Quality Visualizations** - 300 DPI diagrams and graphs
- ✅ **Complete Documentation** - Setup, methodology, presentation scripts

---

## Citation

If you use this code or methodology in your research, please cite:

```
VM State Prediction using GRU-Attention Architecture
Thesis Project, December 2025
Dataset: Azure Public Dataset V1
GitHub: [Your Repository URL]
```

---

## Dataset

**Azure Public Dataset V1**
Source: [https://github.com/Azure/AzurePublicDataset](https://github.com/Azure/AzurePublicDataset)

- **VMs:** 156,031 virtual machines
- **Duration:** 30 days
- **Metrics:** CPU utilization (avg, max, min) at 5-minute intervals
- **Classes:** IDLE (81.6%) vs ACTIVE (18.4%)
- **Threshold:** 5% CPU utilization

---

## Troubleshooting

**Issue:** Out of memory during training
**Solution:** Reduce batch size in training scripts (line 200+)

**Issue:** CUDA not found
**Solution:** Install CUDA-compatible TensorFlow: `pip install tensorflow[and-cuda]`

**Issue:** Dataset not loading
**Solution:** Verify `azure_data/` contains vmtable.csv and vm_cpu_readings files

**For more troubleshooting, see [CONFIGURATION_MANUAL.md](CONFIGURATION_MANUAL.md)**

---

## License

MIT License - Free to use for academic and commercial projects

---

## Contact & Support

For questions about this implementation:
- Open an issue in the repository
- See documentation files for detailed explanations
- Refer to CONFIGURATION_MANUAL.md for setup help

---

**Last Updated:** December 2025
**Status:** ✅ Complete - Ready for thesis submission
