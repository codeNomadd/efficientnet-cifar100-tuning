import unittest
import torch
from model import EfficientNetModel, CIFAR100Dataset, set_seed

class TestEfficientNetModel(unittest.TestCase):
    """Test cases for EfficientNet model and dataset."""
    
    def setUp(self):
        """Set up test environment."""
        set_seed(42)
        self.batch_size = 32
        self.num_classes = 100
        
    def test_model_initialization(self):
        """Test model initialization and basic properties."""
        model = EfficientNetModel(num_classes=self.num_classes)
        
        # Test model is on correct device
        self.assertEqual(next(model.get_model().parameters()).device.type, 
                        'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test output shape
        x = torch.randn(self.batch_size, 3, 224, 224)
        output = model.get_model()(x)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        
    def test_dataset_loading(self):
        """Test dataset loading and transformations."""
        dataset = CIFAR100Dataset(batch_size=self.batch_size)
        train_loader, test_loader = dataset.get_data_loaders()
        
        # Test train loader
        train_batch = next(iter(train_loader))
        self.assertEqual(len(train_batch), 2)  # (images, labels)
        self.assertEqual(train_batch[0].shape, (self.batch_size, 3, 224, 224))
        self.assertEqual(train_batch[1].shape, (self.batch_size,))
        
        # Test test loader
        test_batch = next(iter(test_loader))
        self.assertEqual(len(test_batch), 2)  # (images, labels)
        self.assertEqual(test_batch[0].shape, (self.batch_size, 3, 224, 224))
        self.assertEqual(test_batch[1].shape, (self.batch_size,))

if __name__ == '__main__':
    unittest.main() 