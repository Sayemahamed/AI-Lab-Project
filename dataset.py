import os
import torch
import logging
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CovidDataset(Dataset):
    """Dataset class for COVID-19 chest X-ray images."""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform: Optional[transforms.Compose] = None):
        """
        Initialize the dataset.
        
        Args:
            image_paths: List of paths to image files
            labels: List of corresponding labels (0: Normal, 1: COVID)
            transform: Optional transforms to be applied to images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            tuple: (image, label) where image is a tensor and label is an integer
        """
        try:
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            label = self.labels[idx]

            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            raise

def get_transforms(train: bool = True) -> transforms.Compose:
    """
    Get image transforms for training or validation.
    
    Args:
        train: If True, include data augmentation transforms
        
    Returns:
        transforms.Compose: Composition of transforms
    """
    # Common transforms
    common_transforms = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    if train:
        # Add data augmentation for training
        train_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomAutocontrast(p=0.2),
        ]
        return transforms.Compose(train_transforms + common_transforms)
    
    return transforms.Compose(common_transforms)

def get_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    train_split: float = 0.8,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for training and validation.
    
    Args:
        data_dir: Path to the dataset directory
        batch_size: Number of samples in each batch
        train_split: Fraction of data to use for training
        num_workers: Number of worker processes for data loading
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Verify dataset directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    # Collect all image paths and labels
    image_paths = []
    labels = []
    class_to_idx = {'Normal': 0, 'COVID': 1}
    
    for class_name in class_to_idx.keys():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Class directory not found: {class_dir}")
            
        class_images = [
            os.path.join(class_dir, img_name)
            for img_name in os.listdir(class_dir)
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        image_paths.extend(class_images)
        labels.extend([class_to_idx[class_name]] * len(class_images))
        
        logger.info(f"Found {len(class_images)} images for class {class_name}")

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, labels,
        train_size=train_split,
        random_state=42,
        stratify=labels
    )

    logger.info(f"Training set: {len(X_train)} images")
    logger.info(f"Validation set: {len(X_val)} images")

    # Create datasets with appropriate transforms
    train_dataset = CovidDataset(X_train, y_train, transform=get_transforms(train=True))
    val_dataset = CovidDataset(X_val, y_val, transform=get_transforms(train=False))

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader

def get_class_weights(data_dir: str) -> torch.Tensor:
    """
    Calculate class weights for handling class imbalance.
    
    Args:
        data_dir: Path to the dataset directory
        
    Returns:
        torch.Tensor: Tensor of class weights
    """
    class_counts = []
    for class_name in ['Normal', 'COVID']:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            count = len([f for f in os.listdir(class_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            class_counts.append(count)
    
    # Calculate weights as inverse of class frequencies
    total_samples = sum(class_counts)
    class_weights = [total_samples / (len(class_counts) * count) for count in class_counts]
    
    return torch.FloatTensor(class_weights)

if __name__ == '__main__':
    # Example usage
    try:
        data_dir = 'COVID-19_Radiography_Dataset'
        
        # Get class weights for handling imbalance
        class_weights = get_class_weights(data_dir)
        logger.info(f"Class weights: {class_weights}")
        
        # Get data loaders
        train_loader, val_loader = get_data_loaders(data_dir)
        
        # Print dataset information
        logger.info(f"Number of training batches: {len(train_loader)}")
        logger.info(f"Number of validation batches: {len(val_loader)}")
        
        # Test a batch
        for images, labels in train_loader:
            logger.info(f"Batch shape: {images.shape}")
            logger.info(f"Labels shape: {labels.shape}")
            break
            
    except Exception as e:
        logger.error(f"Error in data loading: {str(e)}")
