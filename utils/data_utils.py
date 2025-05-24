#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for data loading and preprocessing.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, random_split

class Cutout(object):
    """
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
        	# (x,y)表示方形补丁的中心位置
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

def get_dataset(dataset_name, batch_size, num_workers=4, data_dir='./data'):
    """
    Load and prepare datasets for training, validation, and testing.
    
    Args:
        dataset_name (str): Name of the dataset ('cifar10', 'cifar100', 'imagenet')
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    if dataset_name.lower() == 'cifar10':
        return get_cifar10(batch_size, num_workers, data_dir)
    elif dataset_name.lower() == 'cifar100':
        return get_cifar100(batch_size, num_workers, data_dir)
    elif dataset_name.lower() == 'imagenet':
        return get_imagenet(batch_size, num_workers, f'{data_dir}/imagenet')
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

def get_cifar10(batch_size, num_workers=4, data_dir='./data'):
    """
    Load CIFAR-10 dataset.
    
    Args:
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Data transformations
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        Cutout(n_holes=1, length=16),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform)
    
    # Split training set into training and validation
    # train_size = int(0.9 * len(train_dataset))
    # val_size = len(train_dataset) - train_size
    # train_dataset, val_dataset = random_split(
    #     train_dataset, [train_size, val_size], 
    #     generator=torch.Generator().manual_seed(42)
    # )
    
    # Apply validation transform to validation dataset
    # val_dataset.dataset.transform = test_transform
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def get_cifar100(batch_size, num_workers=4, data_dir='./data'):
    """
    Load CIFAR-100 dataset.
    
    Args:
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Data transformations
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_transform)
    
    # Split training set into training and validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply validation transform to validation dataset
    val_dataset.dataset.transform = test_transform
    
    test_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=test_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def get_imagenet(batch_size, num_workers=4, data_path='./data/imagenet'):
    """
    Load ImageNet dataset.
    
    Args:
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of workers for data loading
        data_path (str): Path to ImageNet data
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Data transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.ImageFolder(
        root=f'{data_path}/train', transform=train_transform)
    
    val_dataset = torchvision.datasets.ImageFolder(
        root=f'{data_path}/val', transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    # For ImageNet, we use validation set as test set
    test_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
