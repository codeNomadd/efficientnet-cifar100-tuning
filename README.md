# EfficientNet-B0 CIFAR-100 Fine-tuning

This project implements fine-tuning of EfficientNet-B0 model on the CIFAR-100 dataset using PyTorch. The implementation achieves state-of-the-art results with modern training techniques and careful hyperparameter tuning.

## 🏆 Results

- **Top-1 Accuracy**: 86.31% on CIFAR-100 test set
- **Training Time**: ~2 hours on a single GPU
- **Model Size**: ~29MB

![Training Curves](results/train_v1/plots/accuracy_curve.png)

## 📏 Efficiency Metrics

This project includes full evaluation of the trained EfficientNet-B0 model’s efficiency characteristics. These metrics are useful for understanding deployment feasibility on edge devices such as Raspberry Pi or Jetson Nano.

| Metric                        | Value               | Tool Used     |
|------------------------------|---------------------|---------------|
| MACs (Multiply-Accumulate)   | 413.99 M            | THOP          |
| MACs (ptflops)               | 409.04 M            | ptflops       |
| Parameter Count              | 4.14 M              | THOP/ptflops  |
| Model Size (state_dict)      | 16.15 MB            | torch         |
| Inference Time (CPU avg)     | 37.16 ms            | time.time     |
| **Peak RAM Usage (CPU)**     | 366.56 MB           | tracemalloc   |

✅ These values were recorded using a batch size of 1 and image resolution of 224x224.

### 💡 Hardware Implications

- **Deployment Target**: Raspberry Pi 4 (2GB/4GB) and Jetson Nano-class SBCs
- **Notes**:
  - The peak RAM usage and model size suggest the model can be deployed on edge devices **with 2GB+ RAM**.
  - Inference time on CPU (~37ms) allows near real-time performance (~27 FPS possible).
  - Models with lower parameter counts or quantized variants may further reduce footprint.
  - For microcontrollers (e.g., STM32, Arduino), this model is too large without aggressive compression or re-design (e.g., MobileNetV3 Tiny).

## ✨ Features

- EfficientNet-B0 model fine-tuning
- CIFAR-100 dataset training
- Mixed precision training (FP16)
- Gradient accumulation
- Cosine learning rate scheduling with warm restarts
- Comprehensive training metrics and visualization
- Automatic checkpointing and training resume
- GPU memory optimization
- Automated efficiency evaluation: MACs, Params, Inference Time, RAM usage

## 🛠️ Training Methodology

### Model Architecture
- Base model: EfficientNet-B0 (pretrained on ImageNet)
- Modified classifier for CIFAR-100 (100 classes)
- Dropout rate: 0.2

### Data Augmentation
- Random Resized Crop (224x224)
- Random Horizontal Flip (p=0.5)
- Random Rotation (±15 degrees)
- Color Jitter (brightness, contrast, saturation)
- Random Erasing (p=0.4)
- Normalization using CIFAR-100 statistics

### Training Strategy
- **Learning Rate**: 0.001 with Cosine Annealing Warm Restarts
  - T_0: 5 epochs
  - T_mult: 2
  - Minimum LR: 1e-6
- **Optimizer**: AdamW
  - Weight decay: 1e-4
  - Beta1: 0.9
  - Beta2: 0.999
- **Loss Function**: Cross Entropy with Label Smoothing (0.1)
- **Batch Size**: 64
- **Gradient Accumulation**: 4 steps
- **Training Duration**: 80 epochs
- **Mixed Precision**: FP16 training

## 📋 Requirements

- Python 3.7+
- PyTorch 1.12.1
- CUDA-compatible GPU (recommended)
- 8GB+ GPU memory

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/efficientnet-cifar100-finetuning.git
cd efficientnet-cifar100-finetuning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

To start training:

```bash
python training/train_v1.py
```

The training script will:
- Download CIFAR-100 dataset automatically
- Initialize EfficientNet-B0 model
- Train for 80 epochs with automatic checkpointing
- Save training metrics and visualizations in the `results/train_v1` directory

## 📊 Training Configuration

Key training parameters (configurable in `train_v1.py`):
- Batch size: 64
- Learning rate: 0.001
- Weight decay: 1e-4
- Gradient accumulation steps: 4
- Number of epochs: 80
- Label smoothing: 0.1
- Mixed precision: Enabled
- Warm restarts: T_0=5, T_mult=2
- Efficiency metrics evaluated using THOP, ptflops, and memory profiling
- Efficiency metrics evaluation saved to: `results/efficiency_metrics/efficiency_metrics.txt`
- RAM usage profiling using Python `tracemalloc`

## 📁 Project Structure

```
efficientnet-cifar100-finetuning/
├── data/               # CIFAR-100 dataset
│   └── train_v1/     # Training run results
│       ├── checkpoints/  # Model checkpoints
│       ├── metrics/      # Training metrics
│       ├── plots/        # Training visualizations
│       └── logs/         # Training logs
├── training/          # Training code
│   ├── model.py      # Model and dataset implementation
│   ├── train_v1.py   # Training script
│   └── test_model.py # Model testing
│   └── efficiency_test.py  # Model efficiency measurement script
├── results/
│   └── efficiency_metrics/  # Efficiency evaluation results (e.g., .txt logs of THOP, ptflops, RAM usage)
├── LICENSE           # MIT License
└── requirements.txt  # Project dependencies
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📚 References

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) 
# 🔥 EfficientNet-B0 Fine-Tuning on CIFAR-100

This project demonstrates high-performance fine-tuning of EfficientNet-B0 on the CIFAR-100 dataset using PyTorch. It applies modern training techniques, efficient optimization, and rigorous evaluation to achieve SOTA-level accuracy with compact model size.

---

## 📈 Final Results

- **Top-1 Accuracy**: **86.31%**
- **Top-5 Accuracy**: *N/A*
- **Total Epochs**: 80
- **Training Time**: ~2 hours (Single GPU)
- **Model Size**: ~29MB

![Accuracy Curve](results/train_v1/plots/accuracy_curve.png)
![LR Schedule](results/train_v1/plots/learning_rate_schedule.png)

---

## ✅ Features

- EfficientNet-B0 backbone (ImageNet pretrained)
- Fine-tuning on CIFAR-100 (100 classes)
- Strong augmentations (RandAugment, Erasing, ColorJitter)
- Mixed precision (AMP)
- CosineAnnealingWarmRestarts
- Gradient Accumulation
- Label Smoothing Loss
- Auto-resume from checkpoints
- Full metrics, visualizations, and logs
- Automated efficiency evaluation: MACs, Params, Inference Time, RAM usage

---

## 🧠 Training Strategy

**Architecture**
- Base: `EfficientNet-B0`
- Classifier: Modified to 100 classes
- Dropout: 0.2

**Augmentations**
- Resize to 224x224
- Random Crop + Horizontal Flip
- Rotation ±15°, ColorJitter
- Random Erasing (p=0.4)
- Normalize with CIFAR-100 stats

**Optimization**
- Optimizer: `AdamW`  
- LR Scheduler: Cosine Annealing w/ Warm Restarts  
  - `T_0`: 5 epochs, `T_mult`: 2  
  - `LR_init`: 1e-3 → `LR_min`: 1e-6  
- Loss: CrossEntropy + Label Smoothing (0.1)  
- Batch Size: 64  
- Grad Accumulation: 4 steps  
- Mixed Precision: FP16 (AMP)  
- Epochs: 80  

---

## 💻 Quick Start

### 1. Clone and Install
```bash
git clone https://github.com/yourusername/efficientnet-cifar100-finetuning.git
cd efficientnet-cifar100-finetuning
pip install -r requirements.txt
```

### 2. Train
```bash
python training/train_v1.py
```

---

## ⚙️ Configuration (in `train_v1.py`)

```python
batch_size = 64
lr = 0.001
weight_decay = 1e-4
epochs = 80
label_smoothing = 0.1
accumulation_steps = 4
use_amp = True
```

---

## 📁 Directory Structure

```
efficientnet-cifar100-finetuning/
├── data/
├── results/train_v1/
│   ├── checkpoints/
│   ├── metrics/
│   ├── plots/
│   └── logs/
├── training/
│   ├── model.py
│   ├── train_v1.py
│   └── test_model.py
├── LICENSE
└── requirements.txt
```

---

## 📝 License

MIT License. See [LICENSE](LICENSE) for full terms.

---

## 🙌 Acknowledgements

- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [PyTorch Docs](https://pytorch.org/docs/stable/index.html)
# EfficientNet-B0 for CIFAR-100 (PyTorch)

A compact yet powerful PyTorch pipeline for fine-tuning EfficientNet-B0 on the CIFAR-100 dataset. Designed for academic research and edge AI model development.

---

## 🧪 Performance

- 🎯 **Top-1 Accuracy**: 86.31%
- 🕒 **Training Time**: ~2 hours (NVIDIA GPU)
- 🧠 **Model Size**: 29MB
- 🏋️ **Input Size**: 224×224
- 🔁 **Epochs**: 80

---

## 🧰 Features

- Pretrained **EfficientNet-B0** backbone
- CIFAR-100 fine-tuning at 224x224 resolution
- **Label Smoothing**, **Mixup**, and **RandAugment**
- **CosineAnnealingWarmRestarts** scheduler
- **Gradient Accumulation** for memory-efficient training
- **AMP** (mixed precision) support
- **Resume training** from checkpoints
- Full metric tracking, logging, and plots
- Automated efficiency evaluation: MACs, Params, Inference Time, RAM usage

---

## 🧠 Training Strategy

| Component          | Config                                 |
|--------------------|------------------------------------------|
| Optimizer          | AdamW (lr=0.001, wd=1e-4)                |
| LR Scheduler       | CosineAnnealingWarmRestarts (T_0=5)      |
| Loss               | CrossEntropy with Label Smoothing (0.1) |
| Augmentations      | RandomCrop, Flip, Rotation, Jitter, Erasing |
| Batch Size         | 64                                       |
| Epochs             | 80                                       |
| Mixed Precision    | Enabled (AMP)                            |
| Grad Accumulation  | 4 steps                                  |

---

## 🚀 Quick Start

```bash
git clone https://github.com/yourusername/efficientnet-cifar100-finetuning.git
cd efficientnet-cifar100-finetuning
pip install -r requirements.txt
python training/train_v1.py
```

Results and checkpoints will be saved in `results/train_v1/`.

---

## 📁 File Structure

```
efficientnet-cifar100-finetuning/
├── training/
│   ├── model.py          # Model & dataset
│   ├── train_v1.py       # Main training script
│   └── test_model.py     # Inference / evaluation
├── results/train_v1/     # Checkpoints, logs, metrics, plots
├── requirements.txt
└── LICENSE
```

---

## 📚 References

- [EfficientNet (Tan & Le, 2019)](https://arxiv.org/abs/1905.11946)
- [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [PyTorch Docs](https://pytorch.org/docs/stable/index.html)

---

## 📝 License

This project is licensed under the MIT License.