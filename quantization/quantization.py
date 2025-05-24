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
import platform
import warnings

from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import get_default_qconfig_mapping
from torch.fx.graph_module import  GraphModule


def calib_quant_model(model, calib_dataloader):
    """
    calibrate the model with calib_dataloader
    Args:
        model: input model to be quantized
        calib_dataloader: calibration dataset used to collect quantization parameters
    Return:
        model: calibrated model
    """
    assert isinstance(
        model, GraphModule
    ), "model must be a perpared fx ObservedGraphModule."
    model.eval()
    with torch.inference_mode():
        for inputs, labels in calib_dataloader:
            model(inputs)
    print("calib done.")

def quant_fx(model, test_loader):
    """
    use fx api to quantize the model
    Args:
        model: input model to be quantized
    Return:
        model: quantized model
    """
    model_to_quantize = copy.deepcopy(model)
    model_to_quantize.to("cpu")
    model_to_quantize.eval()
    is_x86 = platform.processor().lower() in ['x86_64', 'x86', 'amd64', 'intel64']
    qconfig = get_default_qconfig_mapping('fbgemm' if is_x86 else 'qnnpack')
    prepared_model = prepare_fx(model_to_quantize, qconfig, (1, 3, 224, 224))
    
    calib_quant_model(prepared_model, test_loader)
    quantized_model = convert_fx(prepared_model)
    return quantized_model

def quantize_model(model, train_loader, method='dynamic', bit_width=8, layers_to_skip=None):
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
    
    # Set model to evaluation mode
    float_model.eval()
    
    try:
        if method == 'static':
            try:
                print("Attempting static quantization...")
                quantized_model = static_quantization(float_model, train_loader, bit_width, layers_to_skip)
                # Test if model can run on one batch
                validate_quantized_model(quantized_model, train_loader)
                return quantized_model
            except Exception as e:
                print(f"Static quantization failed: {str(e)}")
                print("Falling back to dynamic quantization...")
                return dynamic_quantization(float_model, bit_width, layers_to_skip)
        elif method == 'dynamic':
            print("Using dynamic quantization...")
            return dynamic_quantization(float_model, bit_width, layers_to_skip)
        elif method == 'qat':
            try:
                print("Attempting quantization-aware training...")
                quantized_model = quantization_aware_training(float_model, train_loader, bit_width, layers_to_skip)
                # Test if model can run on one batch
                validate_quantized_model(quantized_model, train_loader)
                return quantized_model
            except Exception as e:
                print(f"QAT failed: {str(e)}")
                print("Falling back to dynamic quantization...")
                return dynamic_quantization(float_model, bit_width, layers_to_skip)
        else:
            raise ValueError(f"Quantization method {method} not supported")
    except Exception as e:
        print(f"Quantization error: {str(e)}")
        print("Falling back to dynamic quantization...")
        try:
            return dynamic_quantization(float_model, bit_width, layers_to_skip)
        except Exception as e2:
            print(f"All quantization methods failed: {str(e2)}")
            print("Returning original model...")
            return float_model

def validate_quantized_model(model, train_loader):
    """
    Validate that a quantized model can perform inference on a single batch.
    This helps catch incompatibilities early.
    
    Args:
        model (nn.Module): The quantized model to validate
        train_loader (DataLoader): Training data loader
        
    Returns:
        bool: True if validation succeeds
    """
    model.eval()
    # Get a single batch
    for inputs, _ in train_loader:
        with torch.no_grad():
            # Quantized models typically need to run on CPU
            inputs = inputs.cpu()
            _ = model(inputs)
            break
    return True

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
    # Check if we're on a supported platform
    is_x86 = platform.processor().lower() in ['x86_64', 'x86', 'amd64', 'intel64']
    
    # Ensure model is in eval mode
    model.eval()
    
    # Create a fused version of the model to improve quantization
    fused_model = copy.deepcopy(model)
    fused_model = fused_model.to("cpu")
    
    # Try to fuse common layers like Conv+BN+ReLU
    try:
        # For ResNet models, we need specific fusion patterns
        if hasattr(fused_model, 'layer1'):  # Check if it's a ResNet-like model
            torch.quantization.fuse_modules(fused_model, [['conv1', 'bn1', 'relu']], inplace=True)
            
            for module_name, module in fused_model.named_children():
                if 'layer' in module_name:
                    for basic_block_name, basic_block in module.named_children():
                        torch.quantization.fuse_modules(
                            basic_block, 
                            [['conv1', 'bn1', 'relu'], ['conv2', 'bn2']], 
                            inplace=True
                        )
                        # Handle downsample if present
                        if hasattr(basic_block, 'downsample') and basic_block.downsample is not None:
                            torch.quantization.fuse_modules(
                                basic_block.downsample, 
                                [['0', '1']], 
                                inplace=True
                            )
        print("Model fusion complete")
    except Exception as e:
        print(f"Model fusion failed: {str(e)}")
        print("Continuing with unfused model")
    
    # Set the quantization backend
    backend = 'fbgemm' if is_x86 else 'qnnpack'
    
    try:
        # Try using the specified backend
        qconfig = torch.quantization.get_default_qconfig(backend)
    except Exception as e:
        # Fallback to qnnpack if fbgemm isn't available
        print(f"Quantization backend {backend} not available: {str(e)}")
        try:
            qconfig = torch.quantization.get_default_qconfig('qnnpack')
            backend = 'qnnpack'
        except:
            # Last resort - dynamic quantization
            print("Static quantization backends not available. Falling back to dynamic quantization.")
            return dynamic_quantization(model, bit_width, layers_to_skip)
    
    print(f"Using quantization backend: {backend}")

    # Set quantization configuration
    fused_model.qconfig = qconfig
    
    
    # Prepare model for quantization (inserts observers)
    quantization_model = torch.quantization.prepare(fused_model)
    
    # Calibrate with training data (pass data through to observe range)
    with torch.no_grad():
        for i, (inputs, _) in enumerate(train_loader):
            # Ensure inputs are on CPU for quantization
            inputs = inputs.cpu()
            quantization_model(inputs)
            # Limit calibration to a small number of batches for efficiency
            if i >= 10:  # 10 batches should be enough for calibration
                break
    # Convert to quantized model
    quantized_model = torch.quantization.convert(quantization_model)
    return quantized_model

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
    
    # For bit_width of 8, use qint8 data type
    if bit_width == 8:
        dtype = torch.qint8
    else:
        # For other bit widths, we'll default to 8-bit and issue a warning
        dtype = torch.qint8
        print(f"Bit width {bit_width} not supported for dynamic quantization, using 8-bit instead")
    
    # Define quantization configuration - start with just linear layers
    # which have the best support across platforms
    quantizable_ops = {nn.Linear}
    
    # Apply dynamic quantization with basic configuration
    try:
        print("Applying dynamic quantization to Linear layers...")
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            quantizable_ops,
            dtype=dtype
        )
        return quantized_model
    except Exception as e:
        # If specific bit-width fails, try the default implementation
        print(f"Detailed dynamic quantization failed: {str(e)}. Using default implementation.")
        try:
            print("Attempting simplified dynamic quantization...")
            quantized_model = torch.quantization.quantize_dynamic(model, {nn.Linear})
            return quantized_model
        except Exception as e2:
            print(f"All dynamic quantization methods failed: {str(e2)}")
            print("Returning original model...")
            return model

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
    # Check if we're on a supported platform
    is_x86 = platform.processor().lower() in ['x86_64', 'x86', 'amd64', 'intel64']
    backend = 'fbgemm' if is_x86 else 'qnnpack'
    
    try:
        # Try fusion first - same as static quantization
        fused_model = copy.deepcopy(model)
        
        # Try to fuse common layers like Conv+BN+ReLU
        try:
            # For ResNet models, we need specific fusion patterns
            if hasattr(fused_model, 'layer1'):  # Check if it's a ResNet-like model
                torch.quantization.fuse_modules(fused_model, [['conv1', 'bn1', 'relu']], inplace=True)
                
                for module_name, module in fused_model.named_children():
                    if 'layer' in module_name:
                        for basic_block_name, basic_block in module.named_children():
                            torch.quantization.fuse_modules(
                                basic_block, 
                                [['conv1', 'bn1', 'relu'], ['conv2', 'bn2']], 
                                inplace=True
                            )
                            # Handle downsample if present
                            if basic_block.downsample is not None:
                                torch.quantization.fuse_modules(
                                    basic_block.downsample, 
                                    [['0', '1']], 
                                    inplace=True
                                )
            print("Model fusion complete for QAT")
        except Exception as e:
            print(f"Model fusion failed for QAT: {str(e)}")
            print("Continuing with unfused model")
        
        # Set QAT configuration
        try:
            qconfig = torch.quantization.get_default_qat_qconfig(backend)
        except Exception as e:
            print(f"Could not get QAT config for {backend}: {str(e)}")
            
            try:
                qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
                backend = 'qnnpack'
            except Exception as inner_e:
                print(f"QAT not supported: {str(inner_e)}")
                print("Falling back to dynamic quantization...")
                return dynamic_quantization(model, bit_width, layers_to_skip)
        
        # Set quantization configuration
        fused_model.qconfig = qconfig
        
        # Prepare model for QAT
        qat_model = torch.quantization.prepare_qat(fused_model)
        
        # Make sure we're using CPU for training if device is set to 'cpu'
        # Check CUDA availability
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            device = 'cpu'
            
        if device == 'cpu':
            qat_model = qat_model.to('cpu')
        else:
            try:
                qat_model = qat_model.to(device)
            except:
                print(f"Device {device} not available, using CPU")
                device = 'cpu'
                qat_model = qat_model.to('cpu')
        
        # Train with quantization awareness
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(qat_model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            qat_model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = qat_model(inputs)
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
        qat_model = qat_model.cpu()
        quantized_model = torch.quantization.convert(qat_model.eval())
        
        return quantized_model
        
    except Exception as e:
        print(f"QAT failed: {str(e)}")
        print("Falling back to dynamic quantization...")
        return dynamic_quantization(model, bit_width, layers_to_skip)