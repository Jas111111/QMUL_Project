#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for logging.
"""

import logging
import os
import sys
from datetime import datetime

def setup_logger(log_file=None):
    """
    Set up a logger for the project.
    
    Args:
        log_file (str): Path to log file. If None, logs will only be printed to console.
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger('dnn_optimization')
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def log_experiment_config(logger, config):
    """
    Log experiment configuration.
    
    Args:
        logger (logging.Logger): Logger to use
        config (dict): Experiment configuration
        
    Returns:
        None
    """
    logger.info("Experiment configuration:")
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"{key}:")
            for subkey, subvalue in value.items():
                logger.info(f"  {subkey}: {subvalue}")
        else:
            logger.info(f"{key}: {value}")

def log_optimization_results(logger, original_metrics, optimized_metrics, original_flops, optimized_flops):
    """
    Log optimization results.
    
    Args:
        logger (logging.Logger): Logger to use
        original_metrics (dict): Metrics of the original model
        optimized_metrics (dict): Metrics of the optimized model
        original_flops (int): FLOPs of the original model
        optimized_flops (int): FLOPs of the optimized model
        
    Returns:
        None
    """
    # Compute improvement metrics
    accuracy_change = optimized_metrics['accuracy'] - original_metrics['accuracy']
    inference_time_speedup = original_metrics['inference_time'] / optimized_metrics['inference_time']
    flops_reduction = (original_flops - optimized_flops) / original_flops * 100
    
    # Log results
    logger.info("Optimization results:")
    logger.info(f"Original model - Accuracy: {original_metrics['accuracy']:.2f}%, "
                f"Inference time: {original_metrics['inference_time']*1000:.2f}ms, "
                f"FLOPs: {original_flops/1e6:.2f}M")
    logger.info(f"Optimized model - Accuracy: {optimized_metrics['accuracy']:.2f}%, "
                f"Inference time: {optimized_metrics['inference_time']*1000:.2f}ms, "
                f"FLOPs: {optimized_flops/1e6:.2f}M")
    logger.info(f"Improvement - Accuracy change: {accuracy_change:.2f}%, "
                f"Inference speedup: {inference_time_speedup:.2f}x, "
                f"FLOPs reduction: {flops_reduction:.2f}%")

def create_experiment_dir(base_dir='results'):
    """
    Create a directory for the current experiment.
    
    Args:
        base_dir (str): Base directory for experiments
        
    Returns:
        str: Path to the experiment directory
    """
    # Create base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(base_dir, f'experiment_{timestamp}')
    os.makedirs(experiment_dir)
    
    return experiment_dir
