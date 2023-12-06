import torch
import numpy as np
import random
from torch.utils.data import Subset
from torch.utils.data import random_split
from torch.utils.data import Subset, Dataset

class AugmentedDataset(Dataset):
    def __init__(self, original_dataset, indices, transform=None):
        self.original_dataset = Subset(original_dataset, indices)
        self.transform = transform

    def __len__(self):
        return len(self.original_dataset) 

    def __getitem__(self, idx):
        image, label = self.original_dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def set_seed(seed_value=42):
    torch.manual_seed(seed_value)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)
    random.seed(seed_value)

def save_checkpoint(state, filename):
    torch.save(state, filename)

def load_checkpoint(filepath, model, optimizer):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    training_loss = checkpoint['training_loss']
    training_accuracy = checkpoint['training_accuracy']
    validation_loss = checkpoint['validation_loss']
    validation_accuracy = checkpoint['validation_accuracy']

    return model, optimizer, epoch, training_loss, training_accuracy, validation_loss, validation_accuracy

def predict_image(image_tensor, model, device):
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs.data, 1)
    return predicted.item()

def get_subset(dataset, fraction=0.1):
    subset_size = int(len(dataset) * fraction)
    indices = random.sample(range(len(dataset)), subset_size)
    return Subset(dataset, indices)

def train_valid_test_random_split(dataset, train_percent=0.7, valid_percent=0.15):
    dataset_length = len(dataset)
    train_length = int(dataset_length * train_percent)
    valid_length = int(dataset_length * valid_percent)
    test_length = dataset_length - train_length - valid_length
    return random_split(dataset, [train_length, valid_length, test_length])

def simple_moving_average(data, window=10):
    return np.convolve(data, np.ones(window)/window, mode='valid')