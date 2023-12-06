from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset

from logger import logger
from transforms import normalization_transform, augmentation_transform
from config import data_dir, batch_size
from utils import set_seed, train_valid_test_random_split, AugmentedDataset

# Fix randomness for reproducible results
set_seed()

# Loading dataset
dataset = ImageFolder(root=data_dir, transform=normalization_transform)
num_classes = len(dataset.classes)

# Data splitting
train_data, valid_data, test_data = train_valid_test_random_split(
    dataset, train_percent=0.6, valid_percent=0.2)

# Training data augmentation
augmented_train_data = AugmentedDataset(dataset, train_data.indices, transform=augmentation_transform)
combined_train_data = ConcatDataset([train_data, augmented_train_data])

# Generators lighten memory
train_loader = DataLoader(combined_train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

logger.info("Data imported")
