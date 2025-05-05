#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for optimizing deep neural networks.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import json
from datetime import datetime

from utils.model_utils import load_model, save_model, count_parameters, measure_inference_time
from utils.data_utils import get_dataset
from utils.logger import setup_logger, log_experiment_config, log_optimization_results, create_experiment_dir
from compression.weight_sharing import apply_weight_sharing
from compression.low_rank import apply_low_rank_factorization
from pruning.importance_pruning import prune_model as apply_pruning
from distillation.knowledge_distillation import train_with_distillation
from quantization.quantization import quantize_model, static_quantization, dynamic_quantization, quantization_aware_training
from evaluation.metrics import evaluate_model, count_model_flops, get_model_size, plot_comparison, plot_optimization_comparison
from config import get_config, update_config, load_config_from_file, save_config_to_file

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DNN Optimization')
    
    # General arguments
    parser.add_argument('--config', type=str, default=None, help='Path to configuration file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--log_dir', type=str, default='logs', help='Log directory')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset to use')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='resnet18', help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
    
    # Optimization arguments
    parser.add_argument('--optimization', type=str, default=None, 
                        choices=['weight_sharing', 'low_rank', 'pruning', 'distillation', 'quantization', 'all'],
                        help='Optimization method to apply')
    
    # Weight sharing arguments
    parser.add_argument('--n_clusters', type=int, default=32, help='Number of clusters for weight sharing')
    
    # Low-rank factorization arguments
    parser.add_argument('--rank_percent', type=float, default=0.25, help='Percentage of rank to keep')
    
    # Pruning arguments
    parser.add_argument('--sparsity', type=float, default=0.5, help='Target sparsity for pruning')
    parser.add_argument('--pruning_method', type=str, default='magnitude', 
                        choices=['magnitude', 'structured'],
                        help='Pruning method to use')
    
    # Distillation arguments
    parser.add_argument('--teacher_model', type=str, default='resnet50', help='Teacher model for distillation')
    parser.add_argument('--temperature', type=float, default=4.0, help='Temperature for distillation')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for hard loss in distillation')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    
    # Quantization arguments
    parser.add_argument('--quantization_method', type=str, default='static', 
                        choices=['static', 'dynamic', 'qat'],
                        help='Quantization method to use')
    parser.add_argument('--bit_width', type=int, default=8, help='Bit width for quantization')
    
    # Evaluation arguments
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model')
    parser.add_argument('--compare', action='store_true', help='Compare original and optimized models')
    
    return parser.parse_args()

def get_configuration(args):
    """Get configuration from args and config file."""
    # Start with default config
    config = get_config()
    
    # Update from config file if provided
    if args.config:
        config = load_config_from_file(args.config)
    
    # Update from command line arguments
    updates = {
        'general': {
            'seed': args.seed,
            'device': args.device,
            'log_dir': args.log_dir,
            'results_dir': args.output_dir,
        },
        'dataset': {
            'name': args.dataset,
            'batch_size': args.batch_size,
        },
        'model': {
            'name': args.model,
            'pretrained': args.pretrained,
        },
    }
    
    # Add optimization-specific updates
    if args.optimization:
        if args.optimization == 'weight_sharing' or args.optimization == 'all':
            updates['optimization'] = {
                'compression': {
                    'enabled': True,
                    'method': 'weight_sharing',
                    'weight_sharing': {
                        'n_clusters': args.n_clusters,
                    },
                },
            }
        
        if args.optimization == 'low_rank' or args.optimization == 'all':
            updates['optimization'] = updates.get('optimization', {})
            updates['optimization']['compression'] = {
                'enabled': True,
                'method': 'low_rank',
                'low_rank': {
                    'rank_percent': args.rank_percent,
                },
            }
        
        if args.optimization == 'pruning' or args.optimization == 'all':
            updates['optimization'] = updates.get('optimization', {})
            updates['optimization']['pruning'] = {
                'enabled': True,
                'method': args.pruning_method,
                'sparsity': args.sparsity,
            }
        
        if args.optimization == 'distillation' or args.optimization == 'all':
            updates['optimization'] = updates.get('optimization', {})
            updates['optimization']['distillation'] = {
                'enabled': True,
                'teacher_model': args.teacher_model,
                'student_model': args.model,
                'temperature': args.temperature,
                'alpha': args.alpha,
                'epochs': args.epochs,
                'lr': args.lr,
            }
        
        if args.optimization == 'quantization' or args.optimization == 'all':
            updates['optimization'] = updates.get('optimization', {})
            updates['optimization']['quantization'] = {
                'enabled': True,
                'method': args.quantization_method,
                'bit_width': args.bit_width,
            }
    
    # Update evaluation settings
    if args.evaluate or args.compare:
        updates['evaluation'] = {
            'batch_size': args.batch_size,
            'plot_results': True,
        }
    
    # Update config with command line arguments
    config = update_config(config, updates)
    
    return config

def main():
    """Main function."""
    args = parse_args()
    
    # Get configuration
    config = get_configuration(args)
    
    # Create experiment directory
    experiment_dir = create_experiment_dir(config['general']['results_dir'])
    
    # Setup logger
    log_file = os.path.join(experiment_dir, 'experiment.log')
    logger = setup_logger(log_file)
    
    # Log configuration
    logger.info(f"Starting experiment at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_experiment_config(logger, config)
    
    # Save configuration
    config_file = os.path.join(experiment_dir, 'config.json')
    save_config_to_file(config, config_file)
    logger.info(f"Configuration saved to {config_file}")
    
    # Set random seed
    torch.manual_seed(config['general']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['general']['seed'])
    
    # Set device
    device = torch.device(config['general']['device'] if torch.cuda.is_available() and config['general']['device'] == 'cuda' else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load dataset
    logger.info(f"Loading dataset: {config['dataset']['name']}")
    train_loader, val_loader, test_loader = get_dataset(
        config['dataset']['name'], 
        config['dataset']['batch_size'],
        num_workers=config['dataset']['num_workers'],
        data_dir=config['dataset']['data_dir']
    )
    
    # Load model
    logger.info(f"Loading model: {config['model']['name']}")
    num_classes = 10 if config['dataset']['name'] == 'cifar10' else 100 if config['dataset']['name'] == 'cifar100' else 1000
    model = load_model(config['model']['name'], num_classes=num_classes, pretrained=config['model']['pretrained'])
    model = model.to(device)
    
    # Evaluate original model
    logger.info("Evaluating original model")
    original_metrics = evaluate_model(model, test_loader, device)
    original_flops = count_model_flops(model)
    original_size = get_model_size(model, 'MB')
    original_params = count_parameters(model)
    
    logger.info(f"Original model - Accuracy: {original_metrics['accuracy']:.2f}%, "
                f"Inference time: {original_metrics['inference_time']*1000:.2f}ms, "
                f"FLOPs: {original_flops/1e6:.2f}M, "
                f"Size: {original_size:.2f}MB, "
                f"Parameters: {original_params:,}")
    
    # Dictionary to store all optimized models
    optimized_models = {}
    
    # Apply optimizations
    if config['optimization']['compression']['enabled']:
        method = config['optimization']['compression']['method']
        logger.info(f"Applying compression: {method}")
        
        if method == 'weight_sharing':
            n_clusters = config['optimization']['compression']['weight_sharing']['n_clusters']
            layers_to_skip = config['optimization']['compression']['weight_sharing']['layers_to_skip']
            
            logger.info(f"Weight sharing with {n_clusters} clusters")
            weight_sharing_model = apply_weight_sharing(model, n_clusters=n_clusters, layers_to_skip=layers_to_skip)
            optimized_models['weight_sharing'] = weight_sharing_model
            
            # Save model
            save_model(weight_sharing_model, os.path.join(experiment_dir, f"weight_sharing_{config['model']['name']}.pth"))
            logger.info(f"Weight sharing model saved to {os.path.join(experiment_dir, 'weight_sharing_' + config['model']['name'] + '.pth')}")
        
        elif method == 'low_rank':
            rank_percent = config['optimization']['compression']['low_rank']['rank_percent']
            layers_to_skip = config['optimization']['compression']['low_rank']['layers_to_skip']
            
            logger.info(f"Low-rank factorization with rank percent {rank_percent}")
            low_rank_model = apply_low_rank_factorization(model, rank_percent=rank_percent, layers_to_skip=layers_to_skip)
            optimized_models['low_rank'] = low_rank_model
            
            # Save model
            save_model(low_rank_model, os.path.join(experiment_dir, f"low_rank_{config['model']['name']}.pth"))
            logger.info(f"Low-rank model saved to {os.path.join(experiment_dir, 'low_rank_' + config['model']['name'] + '.pth')}")
    
    if config['optimization']['pruning']['enabled']:
        method = config['optimization']['pruning']['method']
        sparsity = config['optimization']['pruning']['sparsity']
        layers_to_skip = config['optimization']['pruning']['layers_to_skip']
        
        logger.info(f"Applying pruning: {method} with sparsity {sparsity}")
        pruned_model = apply_pruning(model, train_loader=train_loader, val_loader=val_loader, device=device, method=method, amount=sparsity)
        optimized_models['pruning'] = pruned_model
        
        # Save model
        save_model(pruned_model, os.path.join(experiment_dir, f"pruned_{config['model']['name']}.pth"))
        logger.info(f"Pruned model saved to {os.path.join(experiment_dir, 'pruned_' + config['model']['name'] + '.pth')}")
    
    if config['optimization']['distillation']['enabled']:
        teacher_model_name = config['optimization']['distillation']['teacher_model']
        student_model_name = config['optimization']['distillation']['student_model']
        temperature = config['optimization']['distillation']['temperature']
        alpha = config['optimization']['distillation']['alpha']
        epochs = config['optimization']['distillation']['epochs']
        lr = config['optimization']['distillation']['lr']
        
        logger.info(f"Applying knowledge distillation: teacher={teacher_model_name}, student={student_model_name}")
        
        # Load teacher model
        teacher_model = load_model(teacher_model_name, num_classes=num_classes, pretrained=True)
        teacher_model = teacher_model.to(device)
        
        # Load student model
        student_model = load_model(student_model_name, num_classes=num_classes, pretrained=False)
        student_model = student_model.to(device)
        
        # Train with distillation
        distilled_model = train_with_distillation(
            teacher_model=teacher_model,
            student_model=student_model,
            train_loader=train_loader,
            val_loader=val_loader,
            temperature=temperature,
            alpha=alpha,
            epochs=epochs,
            lr=lr,
            device=device
        )
        
        optimized_models['distillation'] = distilled_model
        
        # Save model
        save_model(distilled_model, os.path.join(experiment_dir, f"distilled_{config['model']['name']}.pth"))
        logger.info(f"Distilled model saved to {os.path.join(experiment_dir, 'distilled_' + config['model']['name'] + '.pth')}")
    
    if config['optimization']['quantization']['enabled']:
        method = config['optimization']['quantization']['method']
        bit_width = config['optimization']['quantization']['bit_width']
        layers_to_skip = config['optimization']['quantization']['layers_to_skip']
        
        logger.info(f"Applying quantization: {method} with bit width {bit_width}")
        
        # Apply quantization
        quantized_model = quantize_model(
            model, 
            train_loader=train_loader, 
            method=method, 
            bit_width=bit_width, 
            layers_to_skip=layers_to_skip
        )
        
        optimized_models['quantization'] = quantized_model
        
        # Save model
        save_model(quantized_model, os.path.join(experiment_dir, f"quantized_{config['model']['name']}.pth"))
        logger.info(f"Quantized model saved to {os.path.join(experiment_dir, 'quantized_' + config['model']['name'] + '.pth')}")
    
    # If no optimization was applied
    if not optimized_models:
        logger.info("No optimization applied.")
        optimized_models['original'] = model
    
    # Evaluate and compare models
    if config['evaluation']['plot_results']:
        logger.info("Evaluating and comparing models")
        
        # Add original model to the comparison
        all_models = {'original': model}
        all_models.update(optimized_models)
        
        # Plot comparison of all models
        plot_optimization_comparison(
            all_models, 
            test_loader, 
            device, 
            output_dir=experiment_dir
        )
        
        logger.info(f"Comparison plots saved to {experiment_dir}")
    
    logger.info(f"Experiment completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
