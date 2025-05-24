#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for optimizing deep neural networks.
"""

import os
import copy
import argparse
from datetime import datetime

import torch

from baseline.train import train_baseline_model
from compression.low_rank import apply_low_rank_factorization
from compression.weight_sharing import apply_weight_sharing
from distillation.knowledge_distillation import train_with_distillation
from quantization.quantization import quant_fx, quantize_model
from evaluation.metrics import evaluate_model, count_model_flops, count_parameters, get_model_size, plot_optimization_comparison
from utils.model_utils import load_model, load_checkpoint, save_model
from utils.data_utils import get_dataset
from utils.config import load_config, save_config_to_file
from utils.tools import create_experiment_dir
from utils.logger import setup_logger
from pruning.importance_pruning import prune_model as apply_pruning


def args_parser():
    parser = argparse.ArgumentParser(description="Optimization script with OmegaConf config")
    parser.add_argument("--config", type=str, help="Path to YAML/JSON config file")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="OmegaConf dotlist overrides")
    args = parser.parse_args()
    return args

def get_config():
    args = args_parser()
    config = load_config(args.config, args.overrides)
    return config


def main():
    # Get configuration
    config = get_config()
    
    # Create experiment directory
    experiment_dir = create_experiment_dir(config.general.results_dir)
    
    # Setup logger
    log_file = os.path.join(experiment_dir, "experiment.log")
    logger = setup_logger(log_file)
    
    # Save configuration
    config_file = os.path.join(experiment_dir, "config.yaml")
    save_config_to_file(config, config_file)
    logger.info(f"Configuration saved to {config_file}")
    
    # Set random seed
    torch.manual_seed(config.general.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.general.seed)
        
    # Set device
    device = torch.device(
        config.general.device
        if torch.cuda.is_available() and config.general.device == "cuda"
        else "cpu"
    )
    logger.info(f"Using device: {device}")
    
    # Load dataset
    logger.info(f"Loading dataset: {config.dataset.name}")
    train_loader, val_loader, test_loader = get_dataset(
        config.dataset.name,
        config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        data_dir=config.dataset.data_dir,
    )
    
    # Load model
    logger.info(f"Loading model: {config.model.name}")
    num_classes = (
        10
        if config.dataset.name == "cifar10"
        else 100 if config.dataset.name == "cifar100" else 1000
    )
    model = load_model(
        config.model.name,
        num_classes=num_classes,
        pretrained=config.model.pretrained,
    )
    
    # Train baseline model or load from checkpoint
    if config.model.checkpoint is None:
        model = train_baseline_model(
            model,
            train_loader,
            val_loader,
            device=device,
            epochs=config.train.epochs,
            lr=config.train.lr,
        )
        baseline_model_path = os.path.join(experiment_dir, "{}_baseline.pth".format(config.model.name))
        save_model(model,baseline_model_path)
        logger.info("Saved baseline model: {}".format(baseline_model_path))
    else:
        model = load_checkpoint(model, config.model.checkpoint)
        logger.info("Loaded checkpoint: {}".format(config.model.checkpoint))
    
    # Evaluate original model
    logger.info("Evaluating original model")
    original_metrics = evaluate_model(model, test_loader, device)
    original_flops = count_model_flops(model, device=device)
    original_size = get_model_size(model, "MB")
    original_params = count_parameters(model)

    logger.info(
        f"Original model - Accuracy: {original_metrics['accuracy']:.2f}%, "
        f"Inference time: {original_metrics['inference_time']*1000:.2f}ms, "
        f"FLOPs: {original_flops/1e6:.2f}M, "
        f"Size: {original_size:.2f}MB, "
        f"Parameters: {original_params:,}"
    )
    all_models = {"original": model}
    # Dictionary to store all optimized models
    optimized_models = {}

    # Apply optimizations
    if config.optimization.compression.enabled:
        method = config.optimization.compression.method
        logger.info(f"Applying compression: {method}")

        if method == "weight_sharing":
            n_clusters = config.optimization.compression.weight_sharing.n_clusters
            layers_to_skip = config.optimization.compression.weight_sharing.layers_to_skip

            logger.info(f"Weight sharing with {n_clusters} clusters")
            weight_sharing_model = apply_weight_sharing(
                model, n_clusters=n_clusters, layers_to_skip=layers_to_skip
            )
            optimized_models["weight_sharing"] = weight_sharing_model

            # Save model
            save_model(
                weight_sharing_model,
                os.path.join(
                    experiment_dir, f"weight_sharing_{config.model.name}.pth"
                ),
            )
            logger.info(
                f"Weight sharing model saved to {os.path.join(experiment_dir, 'weight_sharing_' + config.model.name + '.pth')}"
            )
            model = copy.deepcopy(weight_sharing_model)

        elif method == "low_rank":
            rank_percent = config.optimization.compression.low_rank.rank_percent
            layers_to_skip = config.optimization.compression.low_rank.layers_to_skip

            logger.info(f"Low-rank factorization with rank percent {rank_percent}")
            low_rank_model = apply_low_rank_factorization(
                model, rank_percent=rank_percent, layers_to_skip=layers_to_skip
            )
            optimized_models["low_rank"] = low_rank_model

            # Save model
            save_model(
                low_rank_model,
                os.path.join(experiment_dir, f"low_rank_{config.model.name}.pth"),
            )
            logger.info(
                f"Low-rank model saved to {os.path.join(experiment_dir, 'low_rank_' + config.model.name + '.pth')}"
            )
            
            model = copy.deepcopy(low_rank_model)

    if config.optimization.pruning.enabled:
        method = config.optimization.pruning.method
        sparsity = config.optimization.pruning.sparsity
        layers_to_skip = config.optimization.pruning.layers_to_skip

        logger.info(f"Applying pruning: {method} with sparsity {sparsity}")
        pruned_model = apply_pruning(
            model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            method=method,
            amount=sparsity,
            epochs=config.optimization.pruning.epochs,
            lr=config.optimization.pruning.lr,
        )
        optimized_models["pruning"] = pruned_model

        # Save model
        save_model(
            pruned_model,
            os.path.join(experiment_dir, f"pruned_{config.model.name}.pth"),
        )
        logger.info(
            f"Pruned model saved to {os.path.join(experiment_dir, 'pruned_' + config.model.name + '.pth')}"
        )
        
        model = copy.deepcopy(pruned_model)
        
    

    if config.optimization.distillation.enabled:
        teacher_model_name = config.optimization.distillation.teacher_model
        student_model_name = config.optimization.distillation.student_model
        temperature = config.optimization.distillation.temperature
        alpha = config.optimization.distillation.alpha
        epochs = config.optimization.distillation.epochs
        lr = config.optimization.distillation.lr

        logger.info(
            f"Applying knowledge distillation: teacher={teacher_model_name}, student={student_model_name}"
        )

        # Load teacher model with proper handling for ResNet50
        teacher_model = load_model(
            teacher_model_name, num_classes=num_classes, pretrained=True
        )

        teacher_model = teacher_model.to(device)
        if config.optimization.distillation.teacher_model_checkpoint is not None:
            teacher_model = load_checkpoint(
                teacher_model, config.optimization.distillation.teacher_model_checkpoint
            )
            logger.info("Loaded teacher model checkpoint from %s", config.optimization.distillation.teacher_model_checkpoint)

        # Load student model
        student_model = load_model(
            student_model_name, num_classes=num_classes, pretrained=False
        )
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
            device=device,
        )

        optimized_models["distillation"] = distilled_model

        # Save model
        save_model(
            distilled_model,
            os.path.join(
                experiment_dir, f"distilled_{config.model.name}.pth"
            ),
        )
        logger.info(
            f"Distilled model saved to {os.path.join(experiment_dir, 'distilled_' + config.model.name + '.pth')}"
        )
        
        model = copy.deepcopy(distilled_model)
        

    if config.optimization.quantization.enabled:
        method = config.optimization.quantization.method
        bit_width = config.optimization.quantization.bit_width
        layers_to_skip = config.optimization.quantization.layers_to_skip

        logger.info(f"Applying quantization: {method} with bit width {bit_width}")
        quantized_model = quant_fx(model, test_loader)

        optimized_models["quantization"] = quantized_model

        # Save model
        save_model(
            quantized_model,
            os.path.join(
                experiment_dir, f"quantized_{config.model.name}.pth"
            ),
        )
        logger.info(
            f"Quantized model saved to {os.path.join(experiment_dir, 'quantized_' + config.model.name + '.pth')}"
        )

    # If no optimization was applied
    if not optimized_models:
        logger.info("No optimization applied.")
        optimized_models["original"] = model

    # Evaluate and compare models
    if config.evaluation.plot_results:
        logger.info("Evaluating and comparing models")

        # Add original model to the comparison
        
        all_models.update(optimized_models)

        try:
            # Plot comparison of all models with error handling
            plot_optimization_comparison(
                all_models, test_loader, device, output_dir=experiment_dir
            )
            logger.info(f"Comparison plots saved to {experiment_dir}")
        except Exception as e:
            logger.error(f"Error creating plots: {str(e)}")
            logger.info("Continuing without plots")

            # Still try to print comparison metrics
            for name, model in all_models.items():
                try:
                    metrics = evaluate_model(model if name != "quantization" else model.to("cpu"), test_loader, device if name != "quantization" else "cpu")
                    flops = count_model_flops(model if name != "quantization" else model.to("cpu"), device=device if name != "quantization" else torch.device("cpu"), is_quantized=True)
                    size = get_model_size(model if name != "quantization" else model.to("cpu"), "MB")
                    params = count_parameters(model if name != "quantization" else model.to("cpu"))

                    logger.info(
                        f"{name} model - Accuracy: {metrics['accuracy']:.2f}%, "
                        f"Inference time: {metrics['inference_time']*1000:.2f}ms, "
                        f"FLOPs: {flops/1e6:.2f}M, "
                        f"Size: {size:.2f}MB, "
                        f"Parameters: {params:,}"
                    )
                except Exception as e2:
                    logger.error(f"Error evaluating {name} model: {str(e2)}")

    logger.info(
        f"Experiment completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    

if __name__ == "__main__":
    main()