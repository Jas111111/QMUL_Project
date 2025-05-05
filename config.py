#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration file for DNN optimization project.
"""

# Default configuration
DEFAULT_CONFIG = {
    # General settings
    'general': {
        'seed': 42,
        'device': 'cuda',
        'log_dir': 'logs',
        'results_dir': 'results',
    },
    
    # Dataset settings
    'dataset': {
        'name': 'cifar10',  # Options: cifar10, cifar100, imagenet
        'batch_size': 128,
        'num_workers': 4,
        'data_dir': 'data',
    },
    
    # Model settings
    'model': {
        'name': 'resnet18',  # Options: resnet18, resnet50, mobilenet_v2, efficientnet_b0
        'pretrained': True,
        'checkpoint': None,  # Path to model checkpoint
    },
    
    # Optimization settings
    'optimization': {
        # Compression settings
        'compression': {
            'enabled': False,
            'method': 'weight_sharing',  # Options: weight_sharing, low_rank
            'weight_sharing': {
                'n_clusters': 32,
                'layers_to_skip': ['conv1', 'fc'],
            },
            'low_rank': {
                'rank_percent': 0.25,
                'layers_to_skip': ['conv1', 'fc'],
            },
        },
        
        # Pruning settings
        'pruning': {
            'enabled': False,
            'method': 'magnitude',  # Options: magnitude, structured
            'sparsity': 0.5,
            'layers_to_skip': ['conv1', 'fc'],
        },
        
        # Distillation settings
        'distillation': {
            'enabled': False,
            'teacher_model': 'resnet50',
            'student_model': 'resnet18',
            'alpha': 0.5,  # Weight for hard loss
            'temperature': 4.0,  # Temperature for soft loss
            'epochs': 100,
            'lr': 0.01,
        },
        
        # Quantization settings
        'quantization': {
            'enabled': False,
            'method': 'static',  # Options: static, dynamic, qat
            'bit_width': 8,
            'layers_to_skip': ['conv1', 'fc'],
        },
    },
    
    # Evaluation settings
    'evaluation': {
        'batch_size': 128,
        'metrics': ['accuracy', 'inference_time', 'flops', 'size', 'parameters'],
        'plot_results': True,
    },
}

def get_config():
    """
    Get the default configuration.
    
    Returns:
        dict: Default configuration
    """
    return DEFAULT_CONFIG.copy()

def update_config(config, updates):
    """
    Update configuration with new values.
    
    Args:
        config (dict): Original configuration
        updates (dict): New values to update
        
    Returns:
        dict: Updated configuration
    """
    for key, value in updates.items():
        if isinstance(value, dict) and key in config and isinstance(config[key], dict):
            update_config(config[key], value)
        else:
            config[key] = value
    
    return config

def load_config_from_file(file_path):
    """
    Load configuration from a file.
    
    Args:
        file_path (str): Path to configuration file
        
    Returns:
        dict: Loaded configuration
    """
    import json
    import yaml
    
    config = get_config()
    
    # Determine file type and load accordingly
    if file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            updates = json.load(f)
    elif file_path.endswith(('.yaml', '.yml')):
        with open(file_path, 'r') as f:
            import yaml
            updates = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {file_path}")
    
    # Update config with loaded values
    update_config(config, updates)
    
    return config

def save_config_to_file(config, file_path):
    """
    Save configuration to a file.
    
    Args:
        config (dict): Configuration to save
        file_path (str): Path to save configuration to
        
    Returns:
        None
    """
    import json
    import yaml
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Determine file type and save accordingly
    if file_path.endswith('.json'):
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=4)
    elif file_path.endswith(('.yaml', '.yml')):
        with open(file_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        raise ValueError(f"Unsupported configuration file format: {file_path}")
