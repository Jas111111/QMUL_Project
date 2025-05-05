#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of quantization techniques for model compression.
"""

import torch
import torch.nn as nn
import torch.quantization
import copy
import numpy as np

def quantize_model(model, train_loader, method='static', bit_width=8, layers_to_skip=None):
    """
    Quantize a model to reduce memory footprint and improve inference speed.
    
    Args:
        model (nn.Module): The model to quantize
        train_loader (DataLoader): Training data loader for calibration
        method (str): Quantization method ('static', 'dynamic', 'qat')
        bit_width (int): Target bit width (8, 4, etc.)
        layers_to_skip (list): List of layer names to skip
        
    Returns:
        nn.Module: Quantized model
    """
    if layers_to_skip is None:
        layers_to_skip = []
    
    # Create a copy of the model to avoid modifying the original
    float_model = copy.deepcopy(model)
    
    if method == 'static':
        return static_quantization(float_model, train_loader, bit_width, layers_to_skip)
    elif method == 'dynamic':
        return dynamic_quantization(float_model, bit_width, layers_to_skip)
    elif method == 'qat':
        return quantization_aware_training(float_model, train_loader, bit_width, layers_to_skip)
    else:
        raise ValueError(f"Quantization method {method} not supported")

def static_quantization(model, train_loader, bit_width=8, layers_to_skip=None):
    """
    Apply static quantization to a model.
    
    Args:
        model (nn.Module): The model to quantize
        train_loader (DataLoader): Training data loader for calibration
        bit_width (int): Target bit width (8, 4, etc.)
        layers_to_skip (list): List of layer names to skip
        
    Returns:
        nn.Module: Quantized model
    """
    # Prepare model for static quantization
    model.eval()
    
    # Specify quantization configuration
    if bit_width == 8:
        qconfig = torch.quantization.get_default_qconfig('fbgemm')
    elif bit_width == 4:
        # For 4-bit quantization, we need a custom qconfig
        qconfig = torch.quantization.QConfig(
            activation=torch.quantization.MinMaxObserver.with_args(
                qscheme=torch.per_tensor_symmetric, dtype=torch.quint4x2
            ),
            weight=torch.quantization.MinMaxObserver.with_args(
                qscheme=torch.per_tensor_symmetric, dtype=torch.qint4x2
            )
        )
    else:
        raise ValueError(f"Bit width {bit_width} not supported")
    
    # Set quantization configuration
    model.qconfig = qconfig
    
    # Prepare model for quantization
    torch.quantization.prepare(model, inplace=True)
    
    # Calibrate with training data
    with torch.no_grad():
        for inputs, _ in train_loader:
            model(inputs)
    
    # Convert to quantized model
    torch.quantization.convert(model, inplace=True)
    
    return model

def dynamic_quantization(model, bit_width=8, layers_to_skip=None):
    """
    Apply dynamic quantization to a model.
    
    Args:
        model (nn.Module): The model to quantize
        bit_width (int): Target bit width (8, 4, etc.)
        layers_to_skip (list): List of layer names to skip
        
    Returns:
        nn.Module: Quantized model
    """
    # Prepare model for dynamic quantization
    model.eval()
    
    # Define quantization configuration
    if bit_width == 8:
        dtype = torch.qint8
    elif bit_width == 4:
        dtype = torch.qint4x2  # This is a placeholder, actual 4-bit support depends on PyTorch version
    else:
        raise ValueError(f"Bit width {bit_width} not supported")
    
    # Apply dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM, nn.LSTMCell, nn.RNNCell, nn.GRUCell},
        dtype=dtype
    )
    
    return quantized_model

def quantization_aware_training(model, train_loader, bit_width=8, layers_to_skip=None, 
                               epochs=5, lr=0.0001, device='cuda'):
    """
    Apply quantization-aware training (QAT) to a model.
    
    Args:
        model (nn.Module): The model to quantize
        train_loader (DataLoader): Training data loader
        bit_width (int): Target bit width (8, 4, etc.)
        layers_to_skip (list): List of layer names to skip
        epochs (int): Number of QAT epochs
        lr (float): Learning rate for QAT
        device (str): Device to use for training
        
    Returns:
        nn.Module: Quantized model
    """
    # Prepare model for QAT
    model.train()
    
    # Specify quantization configuration
    if bit_width == 8:
        qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    elif bit_width == 4:
        # For 4-bit QAT, we need a custom qconfig
        qconfig = torch.quantization.QConfig(
            activation=torch.quantization.FakeQuantize.with_args(
                observer=torch.quantization.MovingAverageMinMaxObserver,
                quant_min=0, quant_max=15  # 4-bit unsigned
            ),
            weight=torch.quantization.FakeQuantize.with_args(
                observer=torch.quantization.MovingAverageMinMaxObserver,
                quant_min=-8, quant_max=7  # 4-bit signed
            )
        )
    else:
        raise ValueError(f"Bit width {bit_width} not supported")
    
    # Set quantization configuration
    model.qconfig = qconfig
    
    # Prepare model for QAT
    torch.quantization.prepare_qat(model, inplace=True)
    
    # Move model to device
    model = model.to(device)
    
    # Train with quantization awareness
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.2f}%")
    
    # Convert to quantized model
    model = model.cpu()
    torch.quantization.convert(model, inplace=True)
    
    return model

def post_training_quantization(model, calibration_loader, bit_width=8, backend='fbgemm'):
    """
    Apply post-training quantization to a model.
    
    Args:
        model (nn.Module): The model to quantize
        calibration_loader (DataLoader): Data loader for calibration
        bit_width (int): Target bit width (8, 4, etc.)
        backend (str): Backend to use ('fbgemm' for x86, 'qnnpack' for ARM)
        
    Returns:
        nn.Module: Quantized model
    """
    # Set model to evaluation mode
    model.eval()
    
    # Create a copy of the model with modules swapped for quantizable versions
    quantizable_model = torch.quantization.QuantWrapper(copy.deepcopy(model))
    
    # Specify quantization configuration
    if backend == 'fbgemm':
        quantizable_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    else:  # qnnpack
        quantizable_model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    
    # Prepare model for quantization
    prepared_model = torch.quantization.prepare(quantizable_model)
    
    # Calibrate with data
    with torch.no_grad():
        for inputs, _ in calibration_loader:
            prepared_model(inputs)
    
    # Convert to quantized model
    quantized_model = torch.quantization.convert(prepared_model)
    
    return quantized_model

class CustomQuantizedModule(nn.Module):
    """
    Custom module for manual quantization implementation.
    """
    def __init__(self, module, bit_width=8, symmetric=False):
        """
        Initialize the custom quantized module.
        
        Args:
            module (nn.Module): Original module to quantize
            bit_width (int): Target bit width
            symmetric (bool): Whether to use symmetric quantization
        """
        super(CustomQuantizedModule, self).__init__()
        self.original_module = module
        self.bit_width = bit_width
        self.symmetric = symmetric
        
        # Compute quantization parameters
        self.n_levels = 2 ** bit_width
        if symmetric:
            self.min_val = -(self.n_levels // 2)
            self.max_val = (self.n_levels // 2) - 1
        else:
            self.min_val = 0
            self.max_val = self.n_levels - 1
        
        # Initialize scale and zero point
        self.scale = nn.Parameter(torch.ones(1), requires_grad=False)
        self.zero_point = nn.Parameter(torch.zeros(1), requires_grad=False)
        
        # Register hooks for calibration
        self.register_buffer('min_weight', torch.tensor(float('inf')))
        self.register_buffer('max_weight', torch.tensor(float('-inf')))
    
    def update_params(self):
        """Update quantization parameters based on observed min/max values."""
        if self.symmetric:
            max_abs = max(abs(self.min_weight), abs(self.max_weight))
            self.scale.data = (2 * max_abs) / (self.max_val - self.min_val)
            self.zero_point.data = torch.zeros(1)
        else:
            self.scale.data = (self.max_weight - self.min_weight) / (self.max_val - self.min_val)
            self.zero_point.data = torch.round(self.min_val - self.min_weight / self.scale)
    
    def quantize(self, x):
        """Quantize a tensor."""
        x_q = torch.round(x / self.scale + self.zero_point)
        x_q = torch.clamp(x_q, self.min_val, self.max_val)
        return x_q
    
    def dequantize(self, x_q):
        """Dequantize a tensor."""
        return (x_q - self.zero_point) * self.scale
    
    def forward(self, x):
        """Forward pass with quantization/dequantization."""
        if self.training:
            # Update min/max values during training
            with torch.no_grad():
                weight = self.original_module.weight.data
                self.min_weight = torch.min(torch.min(weight), self.min_weight)
                self.max_weight = torch.max(torch.max(weight), self.max_weight)
            
            # Use original module during training
            return self.original_module(x)
        else:
            # Quantize weights
            w_q = self.quantize(self.original_module.weight)
            w = self.dequantize(w_q)
            
            # Use quantized weights for inference
            if isinstance(self.original_module, nn.Linear):
                output = F.linear(
                    x, w, 
                    self.original_module.bias if self.original_module.bias is not None else None
                )
            elif isinstance(self.original_module, nn.Conv2d):
                output = F.conv2d(
                    x, w, 
                    self.original_module.bias if self.original_module.bias is not None else None,
                    self.original_module.stride, self.original_module.padding,
                    self.original_module.dilation, self.original_module.groups
                )
            else:
                raise TypeError(f"Unsupported module type: {type(self.original_module)}")
            
            return output

def manual_quantization(model, bit_width=8, symmetric=True):
    """
    Apply manual quantization to a model.
    
    Args:
        model (nn.Module): The model to quantize
        bit_width (int): Target bit width
        symmetric (bool): Whether to use symmetric quantization
        
    Returns:
        nn.Module: Quantized model
    """
    # Create a copy of the model to avoid modifying the original
    quantized_model = copy.deepcopy(model)
    
    # Replace modules with custom quantized versions
    for name, module in list(quantized_model.named_children()):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            setattr(quantized_model, name, CustomQuantizedModule(module, bit_width, symmetric))
        else:
            # Recursively quantize submodules
            setattr(quantized_model, name, manual_quantization(module, bit_width, symmetric))
    
    return quantized_model
