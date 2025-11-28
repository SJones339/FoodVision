# Custom CNN vs Transfer Learning: Deep Learning Comparison

## Overview

This project now includes **two classification approaches** to demonstrate different deep learning techniques:

1. **Custom CNN (Training from Scratch)**: Our own CNN architecture trained from scratch
2. **EfficientNet (Transfer Learning)**: Pretrained model fine-tuned on food data

This comparison is **perfect for your DL class presentation** because it shows:
- Understanding of CNN architecture design
- Transfer learning vs training from scratch
- Trade-offs between approaches
- Real performance comparisons

---

## Architecture Comparison

### Custom CNN (Our Design)

**Architecture**:
```
Input (224x224x3)
  ↓
Conv Block 1: 64 filters → MaxPool
  ↓
Conv Block 2: 128 filters → MaxPool
  ↓
Conv Block 3: 256 filters → MaxPool
  ↓
Conv Block 4: 512 filters → MaxPool
  ↓
Conv Block 5: 512 filters → MaxPool
  ↓
Global Average Pooling
  ↓
FC1 (512) → Dropout → ReLU
  ↓
FC2 (101 classes) → Output
```

**Key Features**:
- **5 convolutional blocks** with progressive feature extraction
- **Batch normalization** after each conv layer
- **Dropout** for regularization (0.5)
- **Global Average Pooling** to reduce parameters
- **He initialization** for weights (good for ReLU)
- **~15-20M parameters**

**Design Choices**:
- Progressive increase in filters (64 → 512)
- Max pooling for downsampling (reduces spatial dimensions)
- Dropout prevents overfitting
- No pretrained weights - learns everything from food data

### EfficientNet (Transfer Learning)

**Architecture**:
- Pretrained on ImageNet (1.4M images, 1000 classes)
- Compound scaling (depth, width, resolution)
- Mobile inverted bottleneck blocks
- **~5M parameters** (EfficientNet-B0)

**Key Features**:
- **Pretrained weights** from ImageNet
- **Transfer learning**: Only fine-tune last layers
- **Efficient architecture**: Good accuracy/speed trade-off
- **Proven architecture**: State-of-the-art results

---

## Training Comparison

### Custom CNN (From Scratch)

**Training Process**:
1. Initialize weights randomly (He initialization)
2. Train entire network from scratch
3. Learn all features from food data
4. Typically needs more epochs to converge

**Expected Results**:
- **Training Time**: Longer (needs to learn all features)
- **Convergence**: Slower (more epochs needed)
- **Final Accuracy**: Lower (typically 60-75% on Food-101)
- **Data Requirements**: Needs more data to learn features

### EfficientNet (Transfer Learning)

**Training Process**:
1. Load pretrained ImageNet weights
2. Replace classifier head (1000 → 101 classes)
3. Fine-tune entire network (or freeze backbone)
4. Leverages learned features from ImageNet

**Expected Results**:
- **Training Time**: Shorter (faster convergence)
- **Convergence**: Faster (fewer epochs needed)
- **Final Accuracy**: Higher (typically 80-90% on Food-101)
- **Data Requirements**: Works with less data

---

## Key Deep Learning Concepts Demonstrated

### 1. **Transfer Learning**
- **What**: Using pretrained models on new tasks
- **Why**: Leverages features learned on large datasets (ImageNet)
- **How**: Fine-tune pretrained EfficientNet on Food-101
- **Benefit**: Faster training, better accuracy, less data needed

### 2. **Training from Scratch**
- **What**: Learning all features from data
- **Why**: Full control, understand what's learned
- **How**: Custom CNN with random initialization
- **Benefit**: Domain-specific features, interpretability

### 3. **CNN Architecture Design**
- **Convolutional Layers**: Feature extraction
- **Pooling**: Dimensionality reduction
- **Batch Normalization**: Stabilizes training
- **Dropout**: Prevents overfitting
- **Fully Connected**: Classification

### 4. **Optimization Techniques**
- **Learning Rate Scheduling**: Cosine annealing
- **Weight Decay**: L2 regularization
- **Data Augmentation**: Improves generalization
- **Early Stopping**: Prevents overfitting

---

## Expected Results & Metrics

### Performance Comparison

| Metric | Custom CNN | EfficientNet |
|--------|-----------|-------------|
| **Best Val Accuracy** | 60-75% | 80-90% |
| **Training Time** | Longer | Shorter |
| **Epochs to Converge** | 30-50 | 10-20 |
| **Parameters** | ~15-20M | ~5M |
| **Pretrained Weights** | No | Yes (ImageNet) |
| **Data Efficiency** | Needs more | Works with less |

### Why EfficientNet Performs Better

1. **Pretrained Features**: Learned general image features (edges, textures, shapes)
2. **Proven Architecture**: Optimized for image classification
3. **Transfer Learning**: Leverages knowledge from ImageNet
4. **Efficient Design**: Compound scaling balances accuracy/speed

### Why Custom CNN is Still Valuable

1. **Understanding**: See what features are learned
2. **Control**: Full control over architecture
3. **Domain-Specific**: Can optimize for food images
4. **Educational**: Demonstrates CNN fundamentals

---

## How to Run Comparison

### Step 1: Train Custom CNN
```bash
cd src/classification
python train_custom_cnn.py --model-type v1 --epochs 30 --batch-size 32
```

### Step 2: Train EfficientNet
```python
# In notebook or script
from efficientnet_pytorch import EfficientNet
# ... (see notebook for full code)
```

### Step 3: Compare Results
```python
# Use the comparison notebook
notebooks/03_custom_cnn_comparison.ipynb
```

---

## Presentation Talking Points

### For Your DL Class Presentation:

1. **"We implemented two approaches to demonstrate different deep learning techniques"**
   - Custom CNN: Training from scratch
   - EfficientNet: Transfer learning

2. **"Transfer learning significantly outperforms training from scratch"**
   - Show accuracy comparison (EfficientNet: 85% vs Custom: 70%)
   - Explain why: Pretrained features help

3. **"However, custom CNN provides valuable insights"**
   - Full control over architecture
   - Understand what features are learned
   - Educational value

4. **"Key Deep Learning Concepts"**
   - CNN architecture design
   - Transfer learning
   - Optimization techniques
   - Data augmentation

5. **"Trade-offs"**
   - Accuracy vs Training Time
   - Data Requirements
   - Interpretability

---

## Files Created

1. **`src/classification/custom_cnn.py`**: Custom CNN architecture
2. **`src/classification/train_custom_cnn.py`**: Training script
3. **`notebooks/03_custom_cnn_comparison.ipynb`**: Comparison notebook

---

## Next Steps

1. **Train both models** (can run in parallel)
2. **Compare results** using the notebook
3. **Create visualizations** (training curves, accuracy comparison)
4. **Document findings** for presentation

This comparison adds significant depth to your presentation and demonstrates strong understanding of deep learning concepts!

