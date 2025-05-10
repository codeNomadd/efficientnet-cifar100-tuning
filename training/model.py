import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import random
import numpy as np
import os
from typing import Tuple, Optional

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    """Get the best available device (CUDA or CPU).
    
    Returns:
        torch.device: The device to use for training.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device("cpu")
        print("No GPU available, using CPU")
    return device

class EfficientNetModel:
    """EfficientNet-B0 model implementation for CIFAR-100 classification.
    
    This class implements the EfficientNet-B0 model with modifications for CIFAR-100 dataset.
    It uses transfer learning from ImageNet pretrained weights and modifies the classifier
    for 100 classes.
    """
    
    def __init__(self, num_classes: int = 100) -> None:
        """Initialize EfficientNet-B0 model.
        
        Args:
            num_classes (int): Number of output classes. Defaults to 100 for CIFAR-100.
        """
        self.device = get_device()
        self.model = models.efficientnet_b0(pretrained=True)
        
        # Modify the classifier for CIFAR-100
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_features, num_classes)
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
        print(f"Model is on device: {next(self.model.parameters()).device}")
        
    def get_model(self) -> nn.Module:
        """Get the model instance.
        
        Returns:
            nn.Module: The EfficientNet-B0 model.
        """
        return self.model

class CIFAR100Dataset:
    """CIFAR-100 dataset implementation with data augmentation.
    
    This class handles the loading and preprocessing of CIFAR-100 dataset,
    including data augmentation for training and basic preprocessing for testing.
    """
    
    def __init__(self, batch_size: int = 128, num_workers: int = 4) -> None:
        """Initialize CIFAR-100 dataset.
        
        Args:
            batch_size (int): Batch size for data loading. Defaults to 128.
            num_workers (int): Number of workers for data loading. Defaults to 4.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Define transformations for training
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL Image to tensor first
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.3, 0.3, 0.3),
            transforms.RandomErasing(p=0.4, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        ])
        
        # Define transformations for testing
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL Image to tensor first
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        ])
        
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Get train and test data loaders.
        
        Returns:
            Tuple[DataLoader, DataLoader]: Train and test data loaders.
        """
        # Load CIFAR-100 dataset
        train_dataset = datasets.CIFAR100(
            root='./data',
            train=True,
            download=True,
            transform=self.train_transform
        )
        
        test_dataset = datasets.CIFAR100(
            root='./data',
            train=False,
            download=True,
            transform=self.test_transform
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, test_loader 