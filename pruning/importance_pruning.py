#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of model pruning techniques for neural network optimization.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import copy

def magnitude_pruning(model, amount=0.35, layers_to_skip=None):
    """
    Prune model weights based on magnitude.
    
    Args:
        model (nn.Module): The model to prune
        amount (float): Fraction of parameters to prune (0.0-1.0)
        layers_to_skip (list): List of layer names to skip
        
    Returns:
        nn.Module: Pruned model
    """
    if layers_to_skip is None:
        layers_to_skip = []
    
    # Create a copy of the model to avoid modifying the original
    pruned_model = copy.deepcopy(model)
    
    # Apply pruning to each layer
    for name, module in pruned_model.named_modules():
        if name in layers_to_skip:
            continue
            
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            # Make pruning permanent
            prune.remove(module, 'weight')
    
    return pruned_model

def structured_pruning(model, amount=0.3, dim=0, layers_to_skip=None):
    """
    Apply structured pruning to remove entire filters/neurons.
    
    Args:
        model (nn.Module): The model to prune
        amount (float): Fraction of filters/neurons to prune (0.0-1.0)
        dim (int): Dimension along which to prune (0 for output channels/neurons, 1 for input)
        layers_to_skip (list): List of layer names to skip
        
    Returns:
        nn.Module: Pruned model
    """
    if layers_to_skip is None:
        layers_to_skip = []
    
    # Create a copy of the model to avoid modifying the original
    pruned_model = copy.deepcopy(model)
    
    # Apply pruning to each layer
    for name, module in pruned_model.named_modules():
        if name in layers_to_skip:
            continue
            
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=amount, n=2, dim=dim)
            # Make pruning permanent
            prune.remove(module, 'weight')
        elif isinstance(module, nn.Linear):
            prune.ln_structured(module, name='weight', amount=amount, n=2, dim=dim)
            # Make pruning permanent
            prune.remove(module, 'weight')
    
    return pruned_model

def iterative_pruning(model, train_loader, val_loader, device, 
                     initial_amount=0.2, final_amount=0.8, pruning_steps=5, 
                     epochs_per_step=3, lr=0.001):
    """
    Apply iterative pruning with fine-tuning between pruning steps.
    
    Args:
        model (nn.Module): The model to prune
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        device (torch.device): Device to use
        initial_amount (float): Initial pruning amount
        final_amount (float): Final pruning amount
        pruning_steps (int): Number of pruning steps
        epochs_per_step (int): Number of fine-tuning epochs per step
        lr (float): Learning rate for fine-tuning
        
    Returns:
        nn.Module: Pruned and fine-tuned model
    """
    # Create a copy of the model to avoid modifying the original
    current_model = copy.deepcopy(model).to(device)
    
    # Calculate pruning schedule
    pruning_amounts = np.linspace(initial_amount, final_amount, pruning_steps)
    
    # Iterative pruning
    for step, amount in enumerate(pruning_amounts):
        print(f"Pruning step {step+1}/{pruning_steps}, amount: {amount:.2f}")
        
        # Prune the model
        for name, module in current_model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if hasattr(module, 'weight_mask'):
                    # If already pruned, calculate the new amount relative to remaining weights
                    remaining = (module.weight_mask != 0).sum().item()
                    total = module.weight_mask.numel()
                    new_amount = amount * total / remaining
                    new_amount = min(new_amount, 0.9)  # Avoid pruning too much at once
                    
                    # Apply pruning
                    prune.l1_unstructured(module, name='weight', amount=new_amount)
                else:
                    # First pruning
                    prune.l1_unstructured(module, name='weight', amount=amount)
        
        # Fine-tune the model
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(current_model.parameters(), lr=lr)
        
        for epoch in range(epochs_per_step):
            # Training
            current_model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = current_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            train_acc = 100.0 * correct / total
            
            # Validation
            current_model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    outputs = current_model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            val_acc = 100.0 * correct / total
            
            print(f"Epoch {epoch+1}/{epochs_per_step} - Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Train Acc: {train_acc:.2f}%, Val Loss: {val_loss/len(val_loader):.4f}, "
                  f"Val Acc: {val_acc:.2f}%")
    
    # Make pruning permanent
    for name, module in current_model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if hasattr(module, 'weight_mask'):
                prune.remove(module, 'weight')
    
    return current_model

def taylor_pruning(model, train_loader, device, amount=0.3, num_batches=10):
    """
    Prune model weights based on Taylor expansion (gradient-based importance).
    
    Args:
        model (nn.Module): The model to prune
        train_loader (DataLoader): Training data loader
        device (torch.device): Device to use
        amount (float): Fraction of parameters to prune (0.0-1.0)
        num_batches (int): Number of batches to use for gradient computation
        
    Returns:
        nn.Module: Pruned model
    """
    # Create a copy of the model to avoid modifying the original
    pruned_model = copy.deepcopy(model).to(device)
    pruned_model.train()
    
    # Dictionary to store importance scores
    importance_scores = {}
    
    # Register hooks to get gradients
    def hook_fn(name):
        def hook(grad):
            if name in importance_scores:
                importance_scores[name] += (grad * grad).detach().cpu().abs()
            else:
                importance_scores[name] = (grad * grad).detach().cpu().abs()
        return hook
    
    # Register hooks for each parameter
    handles = []
    for name, param in pruned_model.named_parameters():
        if 'weight' in name:
            handle = param.register_hook(hook_fn(name))
            handles.append(handle)
    
    # Compute gradients
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if batch_idx >= num_batches:
            break
            
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = pruned_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Normalize importance scores
    for name in importance_scores:
        importance_scores[name] /= num_batches
    
    # Apply pruning based on importance scores
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            weight_name = f"{name}.weight"
            if weight_name in importance_scores:
                # Create mask based on importance scores
                mask = torch.ones_like(module.weight.data)
                importance = importance_scores[weight_name]
                
                # Flatten for easier processing
                flat_importance = importance.view(-1)
                flat_mask = mask.view(-1)
                
                # Determine threshold
                k = int(flat_importance.numel() * amount)
                if k > 0:
                    threshold = flat_importance.kthvalue(k).values
                    
                    # Create mask
                    flat_mask[flat_importance <= threshold] = 0
                    mask = flat_mask.view_like(module.weight.data)
                    
                    # Apply mask
                    module.weight.data *= mask
    
    return pruned_model

def prune_model(model, train_loader, val_loader, device, method='magnitude', 
               amount=0.35, fine_tune=True, epochs=5, lr=0.001, layers_to_skip=[]):
    """
    Main function to prune a model using different methods.
    
    Args:
        model (nn.Module): The model to prune
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        device (torch.device): Device to use
        method (str): Pruning method ('magnitude', 'structured', 'iterative', 'taylor')
        amount (float): Pruning amount (0.0-1.0)
        fine_tune (bool): Whether to fine-tune after pruning
        epochs (int): Number of fine-tuning epochs
        lr (float): Learning rate for fine-tuning
        
    Returns:
        nn.Module: Pruned model
    """
    if method == 'magnitude':
        pruned_model = magnitude_pruning(model, amount=amount, layers_to_skip=layers_to_skip)
    elif method == 'structured':
        pruned_model = structured_pruning(model, amount=amount, layers_to_skip=layers_to_skip)
    elif method == 'iterative':
        return iterative_pruning(
            model, train_loader, val_loader, device, 
            initial_amount=amount/2, final_amount=amount, 
            pruning_steps=3, epochs_per_step=epochs//3, lr=lr
        )
    elif method == 'taylor':
        pruned_model = taylor_pruning(model, train_loader, device, amount=amount)
    else:
        raise ValueError(f"Pruning method {method} not supported")
    
    # Fine-tune if requested
    if fine_tune:
        pruned_model = pruned_model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(pruned_model.parameters(), lr=lr)
        best_val_acc = 0.0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training
            pruned_model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = pruned_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            train_acc = 100.0 * correct / total
            
            # Validation
            pruned_model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    outputs = pruned_model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            val_acc = 100.0 * correct / total
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = pruned_model.state_dict()
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Train Acc: {train_acc:.2f}%, Val Loss: {val_loss/len(val_loader):.4f}, "
                  f"Val Acc: {val_acc:.2f}%")
            
        if best_model_state:
            pruned_model.load_state_dict(best_model_state)
    
    return pruned_model
