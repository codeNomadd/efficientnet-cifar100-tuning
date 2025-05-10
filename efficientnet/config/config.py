"""
Configuration settings for the EfficientNet training pipeline.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    # Basic training parameters
    seed: int = 42
    batch_size: int = 64
    num_workers: int = 4
    num_epochs: int = 80
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    gradient_accumulation_steps: int = 4
    final_div_factor: float = 10000
    
    # Model parameters
    num_classes: int = 100
    model_name: str = "efficientnet_b0"
    pretrained: bool = True
    
    # Data augmentation parameters
    train_transform: List[Tuple] = None
    test_transform: List[Tuple] = None
    
    # Learning rate scheduler parameters
    scheduler_type: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Mixed precision training
    use_amp: bool = True
    
    # Checkpointing
    save_frequency: int = 5
    checkpoint_dir: str = "checkpoints"
    
    # Logging
    log_frequency: int = 100
    tensorboard_dir: str = "runs"
    
    def __post_init__(self):
        """Initialize default transforms if not provided."""
        if self.train_transform is None:
            self.train_transform = [
                ("random_resized_crop", {"size": 224, "scale": (0.7, 1.0), "ratio": (0.8, 1.2)}),
                ("random_horizontal_flip", {"p": 0.5}),
                ("random_rotation", {"degrees": 15}),
                ("color_jitter", {"brightness": 0.3, "contrast": 0.3, "saturation": 0.3}),
                ("random_erasing", {"p": 0.4, "scale": (0.02, 0.2), "ratio": (0.3, 3.3)}),
                ("normalize", {"mean": [0.5071, 0.4867, 0.4408], "std": [0.2675, 0.2565, 0.2761]})
            ]
        
        if self.test_transform is None:
            self.test_transform = [
                ("resize", {"size": 256}),
                ("center_crop", {"size": 224}),
                ("normalize", {"mean": [0.5071, 0.4867, 0.4408], "std": [0.2675, 0.2565, 0.2761]})
            ]

@dataclass
class ModelConfig:
    """Model configuration parameters."""
    # Model architecture
    model_name: str = "efficientnet_b0"
    num_classes: int = 100
    pretrained: bool = True
    
    # Model optimization
    dropout_rate: float = 0.2
    label_smoothing: float = 0.1
    
    # Model saving
    save_format: str = "pth"
    save_best_only: bool = True
    
    # Model evaluation
    eval_batch_size: int = 128
    num_workers: int = 4

@dataclass
class DataConfig:
    """Data configuration parameters."""
    # Dataset parameters
    dataset_name: str = "CIFAR100"
    data_dir: str = "data"
    train_split: float = 0.9
    
    # Data loading parameters
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    
    # Data augmentation parameters
    use_augmentation: bool = True
    
    # Data normalization
    mean: List[float] = (0.5071, 0.4867, 0.4408)
    std: List[float] = (0.2675, 0.2565, 0.2761)

# Create default configurations
training_config = TrainingConfig()
model_config = ModelConfig()
data_config = DataConfig() 