# EfficientNet-B0 CIFAR-100 Training

This project implements training of EfficientNet-B0 model on the CIFAR-100 dataset using PyTorch. The implementation includes modern training techniques such as mixed precision training, gradient accumulation, and learning rate scheduling.

## Features

- EfficientNet-B0 model implementation
- CIFAR-100 dataset training
- Mixed precision training (FP16)
- Gradient accumulation
- Cosine learning rate scheduling with warm restarts
- Comprehensive training metrics and visualization
- Automatic checkpointing and training resume
- GPU memory optimization

## Requirements

- Python 3.7+
- PyTorch 1.12.1
- CUDA-compatible GPU (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/EfficientNet-dev.git
cd EfficientNet-dev
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

To start training:

```bash
python training/train_v1.py
```

The training script will:
- Download CIFAR-100 dataset automatically
- Initialize EfficientNet-B0 model
- Train for 80 epochs with automatic checkpointing
- Save training metrics and visualizations in the `results/train_v1` directory

## Training Configuration

Key training parameters (configurable in `train_v1.py`):
- Batch size: 64
- Learning rate: 0.001
- Weight decay: 1e-4
- Gradient accumulation steps: 4
- Number of epochs: 80

## Results

Training results are saved in the `results/train_v1` directory:
- Model checkpoints
- Training metrics
- Loss and accuracy plots
- Learning rate schedule visualization
- Model summary

## Project Structure

```
EfficientNet-dev/
├── data/               # CIFAR-100 dataset
├── results/           # Training results and checkpoints
├── training/          # Training code
│   ├── model.py      # Model and dataset implementation
│   └── train_v1.py   # Training script
└── requirements.txt   # Project dependencies
```

## License

[Your chosen license]

## Contributing

Feel free to open issues or submit pull requests for any improvements. 