import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import os
import numpy as np
import gc
import json
import platform
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
from torchinfo import summary
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import signal
import sys

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.model import EfficientNetModel, CIFAR100Dataset, set_seed

# === CONFIG ===
SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = 4
NUM_EPOCHS = 80
LR = 0.001
WEIGHT_DECAY = 1e-4
GRAD_ACCUM_STEPS = 4
FINAL_DIV_FACTOR = 10000

# Create train directory
TRAIN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'train_v1')
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(os.path.join(TRAIN_DIR, 'checkpoints'), exist_ok=True)
os.makedirs(os.path.join(TRAIN_DIR, 'metrics'), exist_ok=True)
os.makedirs(os.path.join(TRAIN_DIR, 'plots'), exist_ok=True)
os.makedirs(os.path.join(TRAIN_DIR, 'logs'), exist_ok=True)

def clear_memory():
    """Clear GPU memory and cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def save_seed_info(seed):
    """Save seed and system information"""
    info = {
        'seed': seed,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    with open(os.path.join(TRAIN_DIR, 'checkpoints', 'seed_info.json'), 'w') as f:
        json.dump(info, f, indent=4)

def save_model_summary(model, batch_size):
    """Save model summary to file"""
    with open(os.path.join(TRAIN_DIR, 'logs', 'model_summary.txt'), 'w') as f:
        f.write(str(summary(model.get_model(), input_size=(batch_size, 3, 224, 224), verbose=0)))

def plot_metrics(train_losses, test_losses, train_accs, test_accs, lrs):
    """Plot training metrics"""
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(TRAIN_DIR, 'plots', 'loss_curve.png'))
    plt.close()
    
    # Plot accuracy curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(os.path.join(TRAIN_DIR, 'plots', 'accuracy_curve.png'))
    plt.close()
    
    # Plot learning rate schedule
    plt.figure(figsize=(10, 5))
    plt.plot(lrs)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.savefig(os.path.join(TRAIN_DIR, 'plots', 'learning_rate_schedule.png'))
    plt.close()

class Trainer:
    def __init__(self, model, train_loader, test_loader, learning_rate=0.001, gradient_accumulation_steps=4):
        """Initialize trainer"""
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = model.device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.best_acc = 0
        self.current_epoch = 0
        self.interrupted = False
        
        # Loss with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            model.get_model().parameters(),
            lr=learning_rate,
            weight_decay=WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=5,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Create directories
        os.makedirs(os.path.join(TRAIN_DIR, 'metrics'), exist_ok=True)
        os.makedirs(os.path.join(TRAIN_DIR, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(TRAIN_DIR, 'checkpoints'), exist_ok=True)
        
        # Initialize metrics file
        with open(os.path.join(TRAIN_DIR, 'metrics', 'training_history.csv'), 'w') as f:
            f.write('epoch,train_loss,test_loss,train_accuracy,test_accuracy,learning_rate,epoch_time\n')
        
        # Initialize metrics lists
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.learning_rates = []

        # Set up signal handler
        signal.signal(signal.SIGINT, self.handle_interrupt)
        signal.signal(signal.SIGTERM, self.handle_interrupt)

    def handle_interrupt(self, signum, frame):
        """Handle interrupt signal"""
        print("\nTraining interrupted! Saving checkpoints...")
        self.interrupted = True
        self.save_checkpoint(self.current_epoch, self.test_losses[-1], self.test_accuracies[-1], is_best=(self.test_accuracies[-1] > self.best_acc))
        print("Checkpoints saved. Exiting...")
        sys.exit(0)

    def train_epoch(self):
        """Train for one epoch"""
        self.model.get_model().train()
        total_loss = 0
        correct = 0
        total = 0
        epoch_start = time.time()
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass with mixed precision
            with autocast():
                outputs = self.model.get_model()(inputs)
                loss = self.criterion(outputs, targets)
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass with gradient accumulation
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.get_model().parameters(), max_norm=1.0)
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            # Update progress bar
            total_loss += loss.item() * self.gradient_accumulation_steps
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': f'{total_loss/(batch_idx+1):.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_time = time.time() - epoch_start
        return total_loss/len(self.train_loader), 100.*correct/total, epoch_time

    def test(self):
        """Test the model"""
        self.model.get_model().eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Testing')
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                with autocast():
                    outputs = self.model.get_model()(inputs)
                    loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                pbar.set_postfix({
                    'loss': f'{total_loss/(batch_idx+1):.3f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        return total_loss/len(self.test_loader), 100.*correct/total, all_preds, all_targets

    def save_checkpoint(self, epoch, test_loss, test_acc, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.get_model().state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': test_loss,
            'accuracy': test_acc,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies,
            'learning_rates': self.learning_rates
        }
        
        if is_best:
            torch.save(checkpoint, os.path.join(TRAIN_DIR, 'checkpoints', 'best_model.pth'))
            self.best_acc = test_acc
            print(f'New best model saved with accuracy: {test_acc:.2f}%')
        elif (epoch + 1) % 5 == 0:  # Save every 5 epochs
            torch.save(checkpoint, os.path.join(TRAIN_DIR, 'checkpoints', f'model_epoch_{epoch+1}.pth'))
            print(f'Checkpoint saved for epoch {epoch+1}')

    def train(self, num_epochs):
        """Train the model"""
        try:
            for epoch in range(num_epochs):
                if self.interrupted:
                    break
                    
                self.current_epoch = epoch
                print(f'\nEpoch: {epoch+1}/{num_epochs}')
                
                # Train
                train_loss, train_acc, epoch_time = self.train_epoch()
                
                # Test
                test_loss, test_acc, preds, targets = self.test()
                
                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Update learning rate
                self.scheduler.step()
                
                # Save metrics to CSV
                with open(os.path.join(TRAIN_DIR, 'metrics', 'training_history.csv'), 'a') as f:
                    f.write(f'{epoch+1},{train_loss:.4f},{test_loss:.4f},{train_acc:.2f},{test_acc:.2f},{current_lr:.6f},{epoch_time:.2f}\n')
                
                # Update metrics lists
                self.train_losses.append(train_loss)
                self.test_losses.append(test_loss)
                self.train_accuracies.append(train_acc)
                self.test_accuracies.append(test_acc)
                self.learning_rates.append(current_lr)
                
                # Update best accuracy and save checkpoint
                if test_acc > self.best_acc:
                    self.best_acc = test_acc
                    print(f'New best test accuracy: {self.best_acc:.2f}%')
                    self.save_checkpoint(epoch, test_loss, test_acc, is_best=True)
                elif (epoch + 1) % 5 == 0:
                    self.save_checkpoint(epoch, test_loss, test_acc)
                
                # Plot metrics
                plot_metrics(
                    self.train_losses,
                    self.test_losses,
                    self.train_accuracies,
                    self.test_accuracies,
                    self.learning_rates
                )
            
            return self.best_acc
            
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            print("Attempting to save checkpoints before exiting...")
            self.save_checkpoint(self.current_epoch, self.test_losses[-1], self.test_accuracies[-1], is_best=(self.test_accuracies[-1] > self.best_acc))
            raise e

def main():
    # Clear memory at start
    clear_memory()
    
    # Set seed for reproducibility
    set_seed(SEED)
    save_seed_info(SEED)
    
    # Initialize model
    print("\nInitializing EfficientNet-B0 model...")
    model = EfficientNetModel(num_classes=100)
    print(f"Model initialized on device: {model.device}")

    # Initialize dataset
    print("\nLoading CIFAR-100 dataset...")
    dataset = CIFAR100Dataset(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    train_loader, test_loader = dataset.get_data_loaders()

    # Save model summary
    save_model_summary(model, batch_size=BATCH_SIZE)

    # Initialize trainer
    print("\nSetting up trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader
    )

    # Train the model
    print("\nStarting training...")
    print(f"Training on device: {model.device}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    
    best_accuracy = trainer.train(NUM_EPOCHS)
    print(f"\nTraining completed! Best accuracy: {best_accuracy:.2f}%")
    
    # Final memory cleanup
    clear_memory()

if __name__ == "__main__":
    main() 