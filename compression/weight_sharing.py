#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of weight sharing techniques for model compression.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
import copy

def apply_weight_sharing(model, n_clusters=16, share_weights=True, layers_to_skip=None):
    """
    Apply weight sharing to a model using K-means clustering.
    
    Args:
        model (nn.Module): The model to compress
        n_clusters (int): Number of clusters for K-means
        share_weights (bool): Whether to actually share weights or just quantize
        layers_to_skip (list): List of layer names to skip
        
    Returns:
        nn.Module: Compressed model with shared weights
    """
    if layers_to_skip is None:
        layers_to_skip = []
    
    # Create a copy of the model to avoid modifying the original
    compressed_model = copy.deepcopy(model)
    
    # Apply weight sharing to each layer
    for name, module in compressed_model.named_modules():
        if name in layers_to_skip:
            continue
            
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            # Get the weights
            weight = module.weight.data.cpu().numpy()
            shape = weight.shape
            
            # Flatten the weights
            flattened_weight = weight.flatten()
            
            # Skip if number of weights is less than number of clusters
            if len(flattened_weight) <= n_clusters:
                continue
                
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(flattened_weight.reshape(-1, 1))
            centroids = kmeans.cluster_centers_.flatten()
            labels = kmeans.labels_
            
            # Create shared weight matrix
            if share_weights:
                # Replace each weight with its corresponding centroid
                compressed_weight = np.choose(labels, centroids).reshape(shape)
                module.weight.data = torch.from_numpy(compressed_weight).to(module.weight.device)
            else:
                # Just for analysis, don't actually modify the weights
                compressed_weight = np.choose(labels, centroids).reshape(shape)
                compression_ratio = len(np.unique(flattened_weight)) / len(np.unique(compressed_weight))
                print(f"Layer {name} compression ratio: {compression_ratio:.2f}x")
    
    return compressed_model

def weight_sharing_with_fine_tuning(model, train_loader, val_loader, device, 
                                    n_clusters=16, epochs=5, lr=0.001):
    """
    Apply weight sharing with fine-tuning to recover accuracy.
    
    Args:
        model (nn.Module): The model to compress
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        device (torch.device): Device to use
        n_clusters (int): Number of clusters for K-means
        epochs (int): Number of fine-tuning epochs
        lr (float): Learning rate for fine-tuning
        
    Returns:
        nn.Module: Compressed and fine-tuned model
    """
    # Apply weight sharing
    compressed_model = apply_weight_sharing(model, n_clusters=n_clusters)
    compressed_model = compressed_model.to(device)
    
    # Fine-tune the model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(compressed_model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        # Training
        compressed_model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = compressed_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_acc = 100.0 * correct / total
        
        # Validation
        compressed_model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = compressed_model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_acc = 100.0 * correct / total
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.2f}%, Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Acc: {val_acc:.2f}%")
    
    return compressed_model

def deep_compression(model, train_loader, val_loader, device, 
                    pruning_threshold=0.1, n_clusters=16, epochs=5, lr=0.001):
    """
    Apply Deep Compression technique (pruning + weight sharing + Huffman coding).
    Based on the paper "Deep Compression: Compressing Deep Neural Networks with Pruning,
    Trained Quantization and Huffman Coding" by Han et al.
    
    Args:
        model (nn.Module): The model to compress
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        device (torch.device): Device to use
        pruning_threshold (float): Threshold for pruning
        n_clusters (int): Number of clusters for weight sharing
        epochs (int): Number of fine-tuning epochs
        lr (float): Learning rate for fine-tuning
        
    Returns:
        nn.Module: Compressed model
    """
    from pruning.importance_pruning import magnitude_pruning
    
    # Step 1: Pruning
    pruned_model = magnitude_pruning(model, pruning_threshold)
    
    # Step 2: Weight sharing (quantization)
    quantized_model = apply_weight_sharing(pruned_model, n_clusters=n_clusters)
    
    # Step 3: Fine-tuning
    compressed_model = weight_sharing_with_fine_tuning(
        quantized_model, train_loader, val_loader, device, 
        n_clusters=n_clusters, epochs=epochs, lr=lr
    )
    
    # Note: Huffman coding is typically applied during model storage, not during runtime
    # We'll skip the actual Huffman coding implementation here
    
    return compressed_model
