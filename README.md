# EfficientNet-B0 CIFAR-100 Fine-tuning

This project implements fine-tuning of EfficientNet-B0 model on the CIFAR-100 dataset using PyTorch. The implementation achieves state-of-the-art results with modern training techniques and careful hyperparameter tuning.

## 🏆 Results

- **Top-1 Accuracy**: 86.31% on CIFAR-100 test set
- **Training Time**: ~2 hours on a single GPU
- **Model Size**: ~29MB

![Training Curves](results/train_v1/plots/accuracy_curve.png)
![Learning Rate Schedule](results/train_v1/plots/learning_rate_schedule.png)

## ✨ Features

- EfficientNet-B0 model fine-tuning
- CIFAR-100 dataset training
- Mixed precision training (FP16)
- Gradient accumulation
- Cosine learning rate scheduling with warm restarts
- Comprehensive training metrics and visualization
- Automatic checkpointing and training resume
- GPU memory optimization

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