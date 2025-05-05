#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of evaluation metrics for model performance analysis.
"""

import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
from thop import profile
import os

def evaluate_model(model, test_loader, device):
    """
    Evaluate a model on test data.
    
    Args:
        model (nn.Module): The model to evaluate
        test_loader (DataLoader): Test data loader
        device (torch.device): Device to use
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()
    model = model.to(device)
    
    # Initialize metrics
    correct = 0
    total = 0
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    # Measure inference time
    inference_times = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Measure inference time
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            
            inference_times.append(end_time - start_time)
            
            # Compute loss
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            # Compute accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    # Compute metrics
    accuracy = 100.0 * correct / total
    avg_loss = test_loss / len(test_loader)
    avg_inference_time = sum(inference_times) / len(inference_times)
    
    # Return metrics
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'inference_time': avg_inference_time,
        'inference_times': inference_times
    }

def count_model_flops(model, input_size=(1, 3, 224, 224)):
    """
    Count the number of FLOPs for a model.
    
    Args:
        model (nn.Module): The model to analyze
        input_size (tuple): Input tensor shape
        
    Returns:
        int: Number of FLOPs
    """
    input_tensor = torch.randn(input_size)
    flops, _ = profile(model, inputs=(input_tensor,))
    return flops

def get_model_size(model, unit='MB'):
    """
    Get the size of a model in bytes, KB, or MB.
    
    Args:
        model (nn.Module): The model to analyze
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

def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model (nn.Module): The model to analyze
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_time(model, input_size=(1, 3, 224, 224), device='cuda', num_runs=100):
    """
    Measure the average inference time of a model.
    
    Args:
        model (nn.Module): The model to analyze
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

def compare_models(original_metrics, optimized_metrics, original_flops, optimized_flops):
    """
    Compare the performance of original and optimized models.
    
    Args:
        original_metrics (dict): Metrics of the original model
        optimized_metrics (dict): Metrics of the optimized model
        original_flops (int): FLOPs of the original model
        optimized_flops (int): FLOPs of the optimized model
        
    Returns:
        dict: Dictionary containing improvement metrics
    """
    # Compute improvement metrics
    accuracy_change = optimized_metrics['accuracy'] - original_metrics['accuracy']
    inference_time_speedup = original_metrics['inference_time'] / optimized_metrics['inference_time']
    flops_reduction = (original_flops - optimized_flops) / original_flops * 100
    
    # Return improvement metrics
    return {
        'accuracy_change': accuracy_change,
        'inference_time_speedup': inference_time_speedup,
        'flops_reduction': flops_reduction
    }

def plot_comparison(original_model, optimized_model, test_loader, device, output_dir='results'):
    """
    Plot comparison between original and optimized models.
    
    Args:
        original_model (nn.Module): Original model
        optimized_model (nn.Module): Optimized model
        test_loader (DataLoader): Test data loader
        device (torch.device): Device to use
        output_dir (str): Directory to save plots
        
    Returns:
        None
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate models
    original_metrics = evaluate_model(original_model, test_loader, device)
    optimized_metrics = evaluate_model(optimized_model, test_loader, device)
    
    # Count FLOPs
    input_size = next(iter(test_loader))[0][0].shape
    input_size = (1,) + input_size
    original_flops = count_model_flops(original_model, input_size)
    optimized_flops = count_model_flops(optimized_model, input_size)
    
    # Get model sizes
    original_size = get_model_size(original_model, 'MB')
    optimized_size = get_model_size(optimized_model, 'MB')
    
    # Count parameters
    original_params = count_parameters(original_model)
    optimized_params = count_parameters(optimized_model)
    
    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.bar(['Original', 'Optimized'], [original_metrics['accuracy'], optimized_metrics['accuracy']])
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy Comparison')
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'))
    plt.close()
    
    # Plot inference time
    plt.figure(figsize=(10, 6))
    plt.bar(['Original', 'Optimized'], [original_metrics['inference_time'], optimized_metrics['inference_time']])
    plt.ylabel('Inference Time (s)')
    plt.title('Model Inference Time Comparison')
    plt.savefig(os.path.join(output_dir, 'inference_time_comparison.png'))
    plt.close()
    
    # Plot FLOPs
    plt.figure(figsize=(10, 6))
    plt.bar(['Original', 'Optimized'], [original_flops / 1e9, optimized_flops / 1e9])
    plt.ylabel('GFLOPs')
    plt.title('Model FLOPs Comparison')
    plt.savefig(os.path.join(output_dir, 'flops_comparison.png'))
    plt.close()
    
    # Plot model size
    plt.figure(figsize=(10, 6))
    plt.bar(['Original', 'Optimized'], [original_size, optimized_size])
    plt.ylabel('Model Size (MB)')
    plt.title('Model Size Comparison')
    plt.savefig(os.path.join(output_dir, 'size_comparison.png'))
    plt.close()
    
    # Plot parameters
    plt.figure(figsize=(10, 6))
    plt.bar(['Original', 'Optimized'], [original_params / 1e6, optimized_params / 1e6])
    plt.ylabel('Parameters (M)')
    plt.title('Model Parameters Comparison')
    plt.savefig(os.path.join(output_dir, 'parameters_comparison.png'))
    plt.close()
    
    # Plot combined metrics
    labels = ['Accuracy (%)', 'Inference Time (s)', 'GFLOPs', 'Size (MB)', 'Parameters (M)']
    original_values = [
        original_metrics['accuracy'],
        original_metrics['inference_time'],
        original_flops / 1e9,
        original_size,
        original_params / 1e6
    ]
    optimized_values = [
        optimized_metrics['accuracy'],
        optimized_metrics['inference_time'],
        optimized_flops / 1e9,
        optimized_size,
        optimized_params / 1e6
    ]
    
    # Normalize values for better visualization
    normalized_original = []
    normalized_optimized = []
    
    for orig, opt in zip(original_values, optimized_values):
        max_val = max(orig, opt)
        normalized_original.append(orig / max_val)
        normalized_optimized.append(opt / max_val)
    
    # Plot radar chart
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    normalized_original += normalized_original[:1]  # Close the loop
    normalized_optimized += normalized_optimized[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.plot(angles, normalized_original, 'b-', linewidth=2, label='Original')
    ax.plot(angles, normalized_optimized, 'r-', linewidth=2, label='Optimized')
    ax.fill(angles, normalized_original, 'b', alpha=0.1)
    ax.fill(angles, normalized_optimized, 'r', alpha=0.1)
    
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title('Model Performance Comparison')
    ax.legend(loc='upper right')
    
    plt.savefig(os.path.join(output_dir, 'radar_comparison.png'))
    plt.close()
    
    # Print comparison summary
    print("\nModel Comparison Summary:")
    print(f"Accuracy: {original_metrics['accuracy']:.2f}% -> {optimized_metrics['accuracy']:.2f}% ({optimized_metrics['accuracy'] - original_metrics['accuracy']:.2f}%)")
    print(f"Inference Time: {original_metrics['inference_time']*1000:.2f}ms -> {optimized_metrics['inference_time']*1000:.2f}ms ({original_metrics['inference_time']/optimized_metrics['inference_time']:.2f}x)")
    print(f"FLOPs: {original_flops/1e6:.2f}M -> {optimized_flops/1e6:.2f}M ({(original_flops - optimized_flops)/original_flops*100:.2f}% reduction)")
    print(f"Model Size: {original_size:.2f}MB -> {optimized_size:.2f}MB ({(original_size - optimized_size)/original_size*100:.2f}% reduction)")
    print(f"Parameters: {original_params/1e6:.2f}M -> {optimized_params/1e6:.2f}M ({(original_params - optimized_params)/original_params*100:.2f}% reduction)")

def plot_optimization_comparison(models_dict, test_loader, device, output_dir='results'):
    """
    Plot comparison between multiple optimization techniques.
    
    Args:
        models_dict (dict): Dictionary mapping model names to model objects
        test_loader (DataLoader): Test data loader
        device (torch.device): Device to use
        output_dir (str): Directory to save plots
        
    Returns:
        None
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Metrics to track
    metrics = {
        'accuracy': [],
        'inference_time': [],
        'flops': [],
        'size': [],
        'parameters': []
    }
    
    # Evaluate each model
    for name, model in models_dict.items():
        print(f"Evaluating {name}...")
        
        # Evaluate model
        model_metrics = evaluate_model(model, test_loader, device)
        
        # Count FLOPs
        input_size = next(iter(test_loader))[0][0].shape
        input_size = (1,) + input_size
        flops = count_model_flops(model, input_size)
        
        # Get model size
        size = get_model_size(model, 'MB')
        
        # Count parameters
        params = count_parameters(model)
        
        # Store metrics
        metrics['accuracy'].append(model_metrics['accuracy'])
        metrics['inference_time'].append(model_metrics['inference_time'])
        metrics['flops'].append(flops / 1e9)  # Convert to GFLOPs
        metrics['size'].append(size)
        metrics['parameters'].append(params / 1e6)  # Convert to M
    
    # Plot metrics
    model_names = list(models_dict.keys())
    
    # Plot accuracy
    plt.figure(figsize=(12, 6))
    plt.bar(model_names, metrics['accuracy'])
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison_all.png'))
    plt.close()
    
    # Plot inference time
    plt.figure(figsize=(12, 6))
    plt.bar(model_names, metrics['inference_time'])
    plt.ylabel('Inference Time (s)')
    plt.title('Model Inference Time Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'inference_time_comparison_all.png'))
    plt.close()
    
    # Plot FLOPs
    plt.figure(figsize=(12, 6))
    plt.bar(model_names, metrics['flops'])
    plt.ylabel('GFLOPs')
    plt.title('Model FLOPs Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'flops_comparison_all.png'))
    plt.close()
    
    # Plot model size
    plt.figure(figsize=(12, 6))
    plt.bar(model_names, metrics['size'])
    plt.ylabel('Model Size (MB)')
    plt.title('Model Size Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'size_comparison_all.png'))
    plt.close()
    
    # Plot parameters
    plt.figure(figsize=(12, 6))
    plt.bar(model_names, metrics['parameters'])
    plt.ylabel('Parameters (M)')
    plt.title('Model Parameters Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameters_comparison_all.png'))
    plt.close()
    
    # Print comparison summary
    print("\nModel Comparison Summary:")
    for i, name in enumerate(model_names):
        print(f"\n{name}:")
        print(f"Accuracy: {metrics['accuracy'][i]:.2f}%")
        print(f"Inference Time: {metrics['inference_time'][i]*1000:.2f}ms")
        print(f"FLOPs: {metrics['flops'][i]:.2f}G")
        print(f"Model Size: {metrics['size'][i]:.2f}MB")
        print(f"Parameters: {metrics['parameters'][i]:.2f}M")
    
    # Create comparison table
    comparison_table = {
        'Model': model_names,
        'Accuracy (%)': metrics['accuracy'],
        'Inference Time (ms)': [t * 1000 for t in metrics['inference_time']],
        'GFLOPs': metrics['flops'],
        'Size (MB)': metrics['size'],
        'Parameters (M)': metrics['parameters']
    }
    
    # Save comparison table as CSV
    import pandas as pd
    df = pd.DataFrame(comparison_table)
    df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    
    # Create HTML table for better visualization
    html_table = df.to_html(index=False)
    with open(os.path.join(output_dir, 'model_comparison.html'), 'w') as f:
        f.write(html_table)
