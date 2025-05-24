#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for model operations like loading, saving, and analyzing models.
"""

import os
import torch
import torch.nn as nn
import torchvision.models as t_models
from torchvision.models import EfficientNet_B0_Weights, MobileNet_V2_Weights
from thop import profile
import time
import utils.model as custom_model

def load_model(model_name, num_classes=1000, pretrained=True):
    """
    Load a model from torchvision models or custom implementations.
    
    Args:
        model_name (str): Name of the model to load
        num_classes (int): Number of output classes
        pretrained (bool): Whether to load pretrained weights
        
    Returns:
        torch.nn.Module: The loaded model
    """
    if model_name == 'resnet18':
        model = custom_model.resnet18()
        if num_classes != 1000:
            print("Loading custom ResNet18")
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'mobilenet_v2':
        model = t_models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        if num_classes != 1000:
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'efficientnet_b0':
        model = t_models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        if num_classes != 1000:
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'squeezenet':
        model = t_models.squeezenet1_1(pretrained=pretrained)
        if num_classes != 1000:
            model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    elif model_name == 'resnet50':
        model = custom_model.resnet50()
        if num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    return model

def save_model(model, path):
    """
    Save a model to disk.
    
    Args:
        model (torch.nn.Module): Model to save
        path (str or Path): Path to save the model
    """
    torch.save(model.state_dict(), path)
    
def load_checkpoint(model, path):
    """
    Load model weights from a checkpoint.
    
    Args:
        model (torch.nn.Module): Model to load weights into
        path (str or Path): Path to the checkpoint
        
    Returns:
        torch.nn.Module: Model with loaded weights
    """
    model.load_state_dict(torch.load(path, weights_only=True))
    return model

def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model (torch.nn.Module): Model to analyze
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_flops(model, input_size=(1, 3, 224, 224)):
    """
    Count the number of FLOPs for a single forward pass.
    
    Args:
        model (torch.nn.Module): Model to analyze
        input_size (tuple): Input tensor shape (batch_size, channels, height, width)
        
    Returns:
        int: Number of FLOPs
    """
    input_tensor = torch.randn(input_size)
    flops = profile(model, inputs=(input_tensor,))[0]
    return flops

def measure_inference_time(model, input_size=(1, 3, 224, 224), device='cuda', num_runs=100):
    """
    Measure the average inference time of a model.
    
    Args:
        model (torch.nn.Module): Model to analyze
        input_size (tuple): Input tensor shape
        device (str): Device to run inference on
        num_runs (int): Number of runs to average over
        
    Returns:
        float: Average inference time in seconds
    """
    model.eval()
    model = model.to(device)
    input_tensor = torch.randn(input_size).to(device)
    
    # Warm-up
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_tensor)
    
    # Measure
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(input_tensor)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    end_time = time.time()
    
    return (end_time - start_time) / num_runs

def get_model_size(model, unit='MB'):
    """
    Get the size of a model in bytes, KB, or MB.
    
    Args:
        model (torch.nn.Module): Model to analyze
        unit (str): Unit to return size in ('B', 'KB', 'MB')
        
    Returns:
        float: Model size in specified unit
    """
    torch.save(model.state_dict(), "temp.p")
    size_bytes = os.path.getsize("temp.p")
    os.remove("temp.p")
    
    if unit == 'KB':
        return size_bytes / 1024
    elif unit == 'MB':
        return size_bytes / (1024 * 1024)
    else:
        return size_bytes
