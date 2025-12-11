# GRU WITH ATTENTION - COMPLETE ARCHITECTURE
## Model: gru_attention_best.h5 (1-Hour Forecasting Model)

---

## ARCHITECTURE SUMMARY

```
Model: "gru_attention"
Total Parameters: 24,162 (94.38 KB)
  - Trainable: 23,970 (93.63 KB)
  - Non-trainable: 192 (768.00 B - BatchNormalization)

Input Shape: (None, 12, 3)
  - 12 timesteps (60 minutes at 5-min intervals)
  - 3 features (avg_cpu, max_cpu, min_cpu)

Output Shape: (None, 1)
  - Binary probability (IDLE vs ACTIVE)
```

---

## LAYER-BY-LAYER ARCHITECTURE

### **1. INPUT LAYER**
```
Layer: input_layer
Type: InputLayer
Output Shape: (None, 12, 3)
Parameters: 0

Purpose: Accepts time series input
Input: 12 timesteps × 3 features (CPU metrics)
```

---

### **2. FIRST GRU LAYER (Encoder)**
```
Layer: gru_2
Type: GRU
Output Shape: (None, 12, 64)
Parameters: 13,248

Configuration:
  - Units: 64 hidden units
  - Return Sequences: True (returns all timestep outputs)
  - Activation: tanh (default)
  - Recurrent Activation: sigmoid (default)

Purpose: Processes temporal dependencies in CPU patterns
Output: 64-dimensional representation for each of 12 timesteps
```

**Mathematical Operations:**
```
For each timestep t:
  Update gate:    z_t = σ(W_z · [h_{t-1}, x_t])
  Reset gate:     r_t = σ(W_r · [h_{t-1}, x_t])
  Candidate:      h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t])
  Hidden state:   h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t

  Where:
    x_t = input at timestep t (3 features)
    h_t = hidden state at timestep t (64 dimensions)
    σ = sigmoid function
    ⊙ = element-wise multiplication
```

---

### **3. BATCH NORMALIZATION (After First GRU)**
```
Layer: batch_normalization
Type: BatchNormalization
Output Shape: (None, 12, 64)
Parameters: 256 (128 trainable + 128 non-trainable)

Purpose: Normalizes activations to stabilize training
Formula: y = γ * (x - μ) / √(σ² + ε) + β
  - μ, σ²: batch mean and variance (non-trainable)
  - γ, β: scale and shift parameters (trainable)
```

---

### **4. DROPOUT (Regularization 1)**
```
Layer: dropout_6
Type: Dropout
Output Shape: (None, 12, 64)
Parameters: 0
Rate: 0.3 (30%)

Purpose: Prevents overfitting by randomly dropping 30% of activations during training
```

---

### **5. SECOND GRU LAYER (Deeper Encoding)**
```
Layer: gru_3
Type: GRU
Output Shape: (None, 12, 32)
Parameters: 9,408

Configuration:
  - Units: 32 hidden units (reduced from 64)
  - Return Sequences: True (returns all timestep outputs for attention)
  - Activation: tanh
  - Recurrent Activation: sigmoid

Purpose: Further processes temporal patterns with dimensionality reduction
Input: 64-dim from previous GRU → Output: 32-dim representations
```

---

### **6. BATCH NORMALIZATION (After Second GRU)**
```
Layer: batch_normalization_1
Type: BatchNormalization
Output Shape: (None, 12, 32)
Parameters: 128 (64 trainable + 64 non-trainable)

Purpose: Normalizes activations from second GRU layer
```

---

### **7. DROPOUT (Regularization 2)**
```
Layer: dropout_7
Type: Dropout
Output Shape: (None, 12, 32)
Parameters: 0
Rate: 0.3 (30%)

Purpose: Additional regularization after second GRU
```

---

## ATTENTION MECHANISM (Layers 8-13)

### **8. ATTENTION SCORE COMPUTATION**
```
Layer: dense
Type: Dense
Output Shape: (None, 12, 1)
Parameters: 33 (32 weights + 1 bias)

Configuration:
  - Units: 1
  - Activation: None (linear)

Purpose: Computes attention score for each timestep
Formula: e_t = W_a · h_t + b_a
  - Input: h_t (32-dim hidden state for each timestep)
  - Output: e_t (scalar attention score)
```

---

### **9. FLATTEN ATTENTION SCORES**
```
Layer: flatten
Type: Flatten
Output Shape: (None, 12)
Parameters: 0

Purpose: Converts (None, 12, 1) → (None, 12) for softmax
Transforms: [[e_1], [e_2], ..., [e_12]] → [e_1, e_2, ..., e_12]
```

---

### **10. SOFTMAX ACTIVATION (Attention Weights)**
```
Layer: activation
Type: Activation
Output Shape: (None, 12)
Parameters: 0
Function: Softmax

Purpose: Converts attention scores to probability distribution
Formula: α_t = exp(e_t) / Σ_i exp(e_i)
  - Σ_t α_t = 1 (attention weights sum to 1)
  - Higher α_t means timestep t is more important
```

**Example Attention Weights:**
```
Timestep:  t-55  t-50  t-45  t-40  t-35  t-30  t-25  t-20  t-15  t-10  t-5   t-0
α_t:       0.04  0.05  0.06  0.07  0.08  0.10  0.12  0.14  0.11  0.09  0.08  0.06
           ↑ Less important (older)              ↑ Most important       ↑ Recent
```

---

### **11. REPEAT ATTENTION WEIGHTS**
```
Layer: repeat_vector
Type: RepeatVector
Output Shape: (None, 32, 12)
Parameters: 0

Purpose: Repeats attention weights 32 times (for 32 GRU units)
Transforms: [α_1, α_2, ..., α_12] → [[α_1, α_2, ..., α_12],
                                      [α_1, α_2, ..., α_12],
                                      ...
                                      [α_1, α_2, ..., α_12]]  (32 times)
```

---

### **12. PERMUTE (Rearrange Dimensions)**
```
Layer: permute
Type: Permute
Output Shape: (None, 12, 32)
Parameters: 0

Purpose: Transposes dimensions to match GRU output shape
Transforms: (None, 32, 12) → (None, 12, 32)
  - Now attention weights align with GRU hidden states
```

---

### **13. MULTIPLY (Apply Attention)**
```
Layer: multiply
Type: Multiply
Output Shape: (None, 12, 32)
Parameters: 0

Purpose: Element-wise multiplication of GRU outputs with attention weights
Formula: h'_t = α_t · h_t
  - Input 1: dropout_7 output (None, 12, 32) - GRU hidden states
  - Input 2: permute output (None, 12, 32) - Attention weights
  - Output: Weighted hidden states (None, 12, 32)

Example:
  If α_5 = 0.15 (high attention), h_5 gets scaled by 0.15 (emphasized)
  If α_1 = 0.03 (low attention), h_1 gets scaled by 0.03 (de-emphasized)
```

---

### **14. GLOBAL AVERAGE POOLING (Context Vector)**
```
Layer: global_average_pooling1d
Type: GlobalAveragePooling1D
Output Shape: (None, 32)
Parameters: 0

Purpose: Aggregates all timesteps into single context vector
Formula: c = (1/12) · Σ_t (α_t · h_t)
  - Averages the attention-weighted hidden states
  - Produces fixed-length representation regardless of sequence length
  - This is the final "context vector" capturing important patterns
```

---

## CLASSIFICATION HEAD (Layers 15-17)

### **15. DENSE LAYER 1 (Feature Processing)**
```
Layer: dense_1
Type: Dense
Output Shape: (None, 32)
Parameters: 1,056 (32×32 weights + 32 bias)

Configuration:
  - Units: 32
  - Activation: ReLU (default)

Purpose: Processes context vector before final prediction
Formula: y = ReLU(W · c + b)
```

---

### **16. DROPOUT (Regularization 3)**
```
Layer: dropout_8
Type: Dropout
Output Shape: (None, 32)
Parameters: 0
Rate: 0.3 (30%)

Purpose: Final regularization before output layer
```

---

### **17. OUTPUT LAYER**
```
Layer: dense_2
Type: Dense
Output Shape: (None, 1)
Parameters: 33 (32 weights + 1 bias)

Configuration:
  - Units: 1
  - Activation: Sigmoid

Purpose: Binary classification (IDLE vs ACTIVE)
Formula: p = σ(W · x + b) = 1 / (1 + e^(-z))
  - Output: Probability ∈ [0, 1]
  - p < threshold → IDLE
  - p ≥ threshold → ACTIVE (threshold = 0.58 for this model)
```

---

## COMPLETE DATA FLOW

```
Input: (batch, 12, 3)
  ↓
[GRU Layer 1: 64 units, return_sequences=True]
  ↓ (batch, 12, 64)
[Batch Normalization]
  ↓ (batch, 12, 64)
[Dropout 30%]
  ↓ (batch, 12, 64)
[GRU Layer 2: 32 units, return_sequences=True]
  ↓ (batch, 12, 32)
[Batch Normalization]
  ↓ (batch, 12, 32)
[Dropout 30%]
  ↓ (batch, 12, 32)  ← GRU outputs (h_t for all timesteps)
        ↓
        ├─────────────────────┐
        ↓                     ↓
[Dense: attention scores]  [Keep for multiplication]
  ↓ (batch, 12, 1)           ↓
[Flatten]                    ↓
  ↓ (batch, 12)              ↓
[Softmax: attention weights] ↓
  ↓ (batch, 12)              ↓
[Repeat 32 times]            ↓
  ↓ (batch, 32, 12)          ↓
[Permute dimensions]         ↓
  ↓ (batch, 12, 32)          ↓
        ↓                     ↓
        └─────────[Multiply]──┘
                  ↓ (batch, 12, 32) ← Attention-weighted states
        [Global Average Pooling]
                  ↓ (batch, 32) ← Context vector
              [Dense 32]
                  ↓ (batch, 32)
              [Dropout 30%]
                  ↓ (batch, 32)
              [Dense 1, Sigmoid]
                  ↓ (batch, 1)
Output: Probability of ACTIVE state
```

---

## PARAMETER BREAKDOWN

```
Component                       Parameters    Percentage
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GRU Layer 1 (64 units)          13,248        54.8%
GRU Layer 2 (32 units)           9,408        38.9%
Batch Normalization                384         1.6%
Attention Mechanism                 33         0.1%
Dense Layer 1 (32 units)         1,056         4.4%
Output Layer (1 unit)               33         0.1%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL                           24,162       100.0%
```

---

## KEY ARCHITECTURAL FEATURES

### 1. **Stacked GRU Architecture**
- **Layer 1 (64 units)**: Captures temporal patterns in raw CPU data
- **Layer 2 (32 units)**: Refines patterns with dimensionality reduction
- **Both return sequences**: Enables attention mechanism to see all timesteps

### 2. **Attention Mechanism**
- **Type**: Additive attention (Bahdanau-style)
- **Purpose**: Learns which timesteps are most predictive
- **Implementation**:
  - Computes attention scores for each timestep
  - Converts to probability distribution (softmax)
  - Applies weights to GRU outputs
  - Aggregates into context vector

### 3. **Regularization Strategy**
- **Dropout (30%)**: Applied after both GRUs and final dense layer
- **Batch Normalization**: Stabilizes training, acts as implicit regularization
- **Combined effect**: Prevents overfitting while maintaining capacity

### 4. **Why This Architecture Works**

**Advantages for 1-Hour Forecasting:**
1. **Temporal Hierarchies**: Two GRU layers capture patterns at different time scales
2. **Attention Focus**: Model learns which historical periods matter most for prediction
3. **Efficient**: Only 24K parameters (smaller than BiGRU's 164K)
4. **Interpretable**: Attention weights reveal which timesteps drive predictions

**Performance on Forecasting Task:**
```
Accuracy: 93.65%
AUC: 96.59%
F1-Macro: 90.89%
Recall (ACTIVE): 98.59% (only 1 active VM missed!)
```

---

## COMPARISON WITH OTHER ARCHITECTURES

```
Architecture          Parameters   Accuracy   F1-Macro   Recall-ACTIVE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Simple GRU               92,545      91.16%     86.68%      84.51%
BiGRU                   164,545      93.37%     90.46%      97.18%
GRU-Attention ⭐        24,162      93.65%     90.89%      98.59%
CNN-GRU                 106,849      93.37%     90.46%      97.18%
CNN-GRU-Attention       110,017      92.82%     89.75%      97.18%
```

**Key Insight**: GRU-Attention achieves **best performance** with **fewest parameters**!

---

## INFERENCE EXAMPLE

```python
import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model('gru_attention_best.h5')

# Prepare input: last 60 minutes of CPU data
# Shape: (1, 12, 3) = 1 sample, 12 timesteps, 3 features
input_data = np.array([[
    [0.02, 0.03, 0.01],  # t-55 min
    [0.02, 0.03, 0.01],  # t-50 min
    [0.03, 0.04, 0.02],  # t-45 min
    [0.03, 0.05, 0.02],  # t-40 min
    [0.04, 0.06, 0.03],  # t-35 min
    [0.05, 0.07, 0.03],  # t-30 min
    [0.06, 0.08, 0.04],  # t-25 min
    [0.07, 0.09, 0.05],  # t-20 min
    [0.08, 0.10, 0.06],  # t-15 min
    [0.09, 0.11, 0.07],  # t-10 min
    [0.10, 0.12, 0.08],  # t-5 min
    [0.11, 0.13, 0.09],  # t-0 min (now)
]])

# Predict state 1 hour from now
probability = model.predict(input_data)[0][0]
optimal_threshold = 0.58
predicted_state = 'ACTIVE' if probability > optimal_threshold else 'IDLE'

print(f"Probability of ACTIVE in 1 hour: {probability:.2%}")
print(f"Predicted State (1 hour ahead): {predicted_state}")
```

**Example Output:**
```
Probability of ACTIVE in 1 hour: 72.45%
Predicted State (1 hour ahead): ACTIVE
```

---

## TRAINING CONFIGURATION

```python
Optimizer: Adam (learning_rate=0.001)
Loss Function: FocalLoss(gamma=2.0, alpha=0.7)
Epochs: 25 (with early stopping)
Batch Size: 64
Class Weights: {0: 0.61, 1: 2.87}  # IDLE: 0.61, ACTIVE: 2.87

Callbacks:
  - EarlyStopping(patience=5, restore_best_weights=True)
  - ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)

Regularization:
  - Dropout: 30% (3 layers)
  - Batch Normalization: 2 layers
  - L2 regularization: None (dropout sufficient)
```

---

## ATTENTION MECHANISM VISUALIZATION

**Conceptual Example of Learned Attention Weights:**

```
Scenario: VM gradually increasing CPU usage

Timestep    CPU%    Attention Weight    Interpretation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
t-55        2.1%         0.03          Low attention (old data)
t-50        2.3%         0.04
t-45        2.8%         0.05
t-40        3.2%         0.06
t-35        3.8%         0.07          Increasing attention
t-30        4.5%         0.09          (upward trend detected)
t-25        5.1%         0.12
t-20        5.8%         0.15          HIGH - crosses threshold
t-15        6.2%         0.14          Model focuses here
t-10        5.9%         0.11
t-5         5.5%         0.08          Recent decline
t-0         5.2%         0.06          Current state
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                      Σ = 1.00

Prediction: ACTIVE (p=0.78)
Reasoning: High attention on t-20 to t-15 where CPU crossed 5% threshold,
           indicating transition to ACTIVE state will persist
```

---

## WHY THIS ARCHITECTURE IS OPTIMAL FOR FORECASTING

1. **Attention learns predictive patterns**: Unlike BiGRU which treats all timesteps equally, attention identifies which historical moments matter most for future prediction

2. **Stacked GRUs capture hierarchies**: First GRU learns low-level patterns (spikes, drops), second GRU learns high-level trends (sustained increase, oscillation)

3. **Efficient parameter usage**: 24K parameters vs 164K (BiGRU) achieves better performance through focused attention rather than brute force

4. **Batch normalization stabilizes**: Deep stacked RNNs can suffer from vanishing/exploding gradients; batch norm prevents this

5. **Targeted regularization**: 30% dropout after each major component prevents overfitting without excessive parameter reduction

---

## SUMMARY

The GRU-Attention model is the **best performing architecture** for 1-hour ahead VM state forecasting:

- **Highest Accuracy**: 93.65%
- **Best Minority Class Recall**: 98.59% (only 1 ACTIVE VM missed)
- **Most Efficient**: 24K parameters (7× smaller than BiGRU)
- **Interpretable**: Attention weights reveal decision process

This architecture successfully combines:
- Temporal modeling (stacked GRUs)
- Selective focus (attention mechanism)
- Robust training (batch norm + dropout)
- Imbalance handling (focal loss + class weights)

The result is a production-ready model for proactive VM consolidation in cloud datacenters.
