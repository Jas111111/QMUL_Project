#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of low-rank factorization techniques for model compression.
"""

import torch
import torch.nn as nn
import numpy as np
import copy
from scipy.linalg import svd

def apply_low_rank_factorization(model, rank_percent=0.25, layers_to_skip=None):
    """
    Apply low-rank factorization to convolutional and fully connected layers.
    
    Args:
        model (nn.Module): The model to compress
        rank_percent (float): Percentage of original rank to keep (0.0-1.0)
        layers_to_skip (list): List of layer names to skip
        
    Returns:
        nn.Module: Compressed model with factorized layers
    """
    if layers_to_skip is None:
        layers_to_skip = []
    
    # Create a copy of the model to avoid modifying the original
    compressed_model = copy.deepcopy(model)
    
    # Dictionary to store the new modules
    new_modules = {}
    
    # Apply low-rank factorization to each layer
    for name, module in compressed_model.named_modules():
        if name in layers_to_skip:
            continue
            
        if isinstance(module, nn.Linear):
            # Get the weights
            weight = module.weight.data.cpu().numpy()
            bias = module.bias.data if module.bias is not None else None
            
            # Compute SVD
            U, S, Vt = svd(weight, full_matrices=False)
            
            # Determine rank to keep
            original_rank = min(weight.shape)
            target_rank = max(1, int(original_rank * rank_percent))
            
            # Create factorized layers
            U_truncated = U[:, :target_rank]
            S_truncated = S[:target_rank]
            Vt_truncated = Vt[:target_rank, :]
            
            # Create new layers
            fc1 = nn.Linear(weight.shape[1], target_rank, bias=False)
            fc2 = nn.Linear(target_rank, weight.shape[0], bias=True if bias is not None else False)
            
            # Set weights
            fc1.weight.data = torch.from_numpy(Vt_truncated).float().to(module.weight.device)
            fc2.weight.data = torch.from_numpy(U_truncated * S_truncated[np.newaxis, :]).float().to(module.weight.device)
            
            if bias is not None:
                fc2.bias.data = torch.from_numpy(bias).float().to(module.weight.device)
            
            # Create sequential module
            factorized_module = nn.Sequential(fc1, fc2)
            
            # Store the new module
            new_modules[name] = factorized_module
            
        elif isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
            # Get the weights
            weight = module.weight.data.cpu().numpy()
            bias = module.bias.data if module.bias is not None else None
            
            # Reshape weight for factorization
            original_shape = weight.shape
            weight_reshaped = weight.reshape(original_shape[0], -1)
            
            # Compute SVD
            U, S, Vt = svd(weight_reshaped, full_matrices=False)
            
            # Determine rank to keep
            original_rank = min(weight_reshaped.shape)
            target_rank = max(1, int(original_rank * rank_percent))
            
            # Create factorized layers
            U_truncated = U[:, :target_rank]
            S_truncated = S[:target_rank]
            Vt_truncated = Vt[:target_rank, :]
            
            # Reshape Vt back to convolutional format
            Vt_conv_shape = (target_rank, original_shape[1], original_shape[2], original_shape[3])
            Vt_conv = Vt_truncated.reshape(Vt_conv_shape)
            
            # Create new layers
            conv1 = nn.Conv2d(
                original_shape[1], target_rank, 
                kernel_size=module.kernel_size, 
                stride=module.stride, 
                padding=module.padding, 
                dilation=module.dilation, 
                groups=module.groups, 
                bias=False
            )
            
            conv2 = nn.Conv2d(
                target_rank, original_shape[0], 
                kernel_size=1, 
                stride=1, 
                padding=0, 
                bias=True if bias is not None else False
            )
            
            # Set weights
            conv1.weight.data = torch.from_numpy(Vt_conv).float().to(module.weight.device)
            
            # Reshape U*S for the pointwise convolution
            US = U_truncated * S_truncated[np.newaxis, :]
            US_conv = US.reshape(original_shape[0], target_rank, 1, 1)
            
            conv2.weight.data = torch.from_numpy(US_conv).float().to(module.weight.device)
            
            if bias is not None:
                conv2.bias.data = torch.from_numpy(bias).float().to(module.weight.device)
            
            # Create sequential module
            factorized_module = nn.Sequential(conv1, conv2)
            
            # Store the new module
            new_modules[name] = factorized_module
    
    # Replace the original modules with factorized ones
    for name, module in new_modules.items():
        # Split the name into parts to navigate the model hierarchy
        parts = name.split('.')
        parent_name = '.'.join(parts[:-1])
        child_name = parts[-1]
        
        if parent_name:
            parent = compressed_model
            for part in parent_name.split('.'):
                parent = getattr(parent, part)
            setattr(parent, child_name, module)
        else:
            setattr(compressed_model, child_name, module)
    
    return compressed_model

def tucker_decomposition_conv_layer(layer, ranks):
    """
    Perform Tucker decomposition on a convolutional layer.
    
    Args:
        layer (nn.Conv2d): Convolutional layer to decompose
        ranks (tuple): Target ranks for input and output channels
        
    Returns:
        nn.Sequential: Decomposed layer
    """
    from tensorly.decomposition import tucker
    import tensorly as tl
    tl.set_backend('pytorch')
    
    # Get layer weights and convert to tensor
    weights = layer.weight.data
    
    # Perform Tucker decomposition
    core, factors = tucker(weights, ranks=ranks)
    
    # Extract factors
    last, first = factors
    
    # Create new layers
    first_layer = nn.Conv2d(
        in_channels=first.shape[0],
        out_channels=first.shape[1],
        kernel_size=1,
        stride=1,
        padding=0,
        bias=False
    )
    
    core_layer = nn.Conv2d(
        in_channels=core.shape[1],
        out_channels=core.shape[0],
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        bias=False
    )
    
    last_layer = nn.Conv2d(
        in_channels=last.shape[1],
        out_channels=last.shape[0],
        kernel_size=1,
        stride=1,
        padding=0,
        bias=layer.bias is not None
    )
    
    # Set weights
    first_layer.weight.data = torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = core
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    
    if layer.bias is not None:
        last_layer.bias.data = layer.bias.data
    
    # Create sequential module
    new_layers = nn.Sequential(first_layer, core_layer, last_layer)
    
    return new_layers

def cp_decomposition_conv_layer(layer, rank):
    """
    Perform CP decomposition on a convolutional layer.
    
    Args:
        layer (nn.Conv2d): Convolutional layer to decompose
        rank (int): Target rank
        
    Returns:
        nn.Sequential: Decomposed layer
    """
    from tensorly.decomposition import parafac
    import tensorly as tl
    tl.set_backend('pytorch')
    
    # Get layer weights and convert to tensor
    weights = layer.weight.data
    
    # Perform CP decomposition
    factors = parafac(weights, rank=rank)
    
    # Extract factors
    last, first, vertical, horizontal = factors
    
    # Create new layers
    first_layer = nn.Conv2d(
        in_channels=first.shape[0],
        out_channels=rank,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=False
    )
    
    vertical_layer = nn.Conv2d(
        in_channels=rank,
        out_channels=rank,
        kernel_size=(layer.kernel_size[0], 1),
        stride=(layer.stride[0], 1),
        padding=(layer.padding[0], 0),
        bias=False,
        groups=rank
    )
    
    horizontal_layer = nn.Conv2d(
        in_channels=rank,
        out_channels=rank,
        kernel_size=(1, layer.kernel_size[1]),
        stride=(1, layer.stride[1]),
        padding=(0, layer.padding[1]),
        bias=False,
        groups=rank
    )
    
    last_layer = nn.Conv2d(
        in_channels=rank,
        out_channels=last.shape[0],
        kernel_size=1,
        stride=1,
        padding=0,
        bias=layer.bias is not None
    )
    
    # Set weights
    first_layer.weight.data = torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    vertical_layer.weight.data = vertical.view(rank, 1, layer.kernel_size[0], 1)
    horizontal_layer.weight.data = horizontal.view(rank, 1, 1, layer.kernel_size[1])
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    
    if layer.bias is not None:
        last_layer.bias.data = layer.bias.data
    
    # Create sequential module
    new_layers = nn.Sequential(first_layer, vertical_layer, horizontal_layer, last_layer)
    
    return new_layers
