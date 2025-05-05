# Deep Neural Network Optimization

This project implements various techniques for optimizing deep neural networks to improve inference speed and reduce model size while maintaining accuracy. The techniques include network compression, model pruning, knowledge distillation, and quantization.

## Project Structure

```
dnn_optimization/
├── compression/
│   ├── weight_sharing.py    # Weight sharing implementation using K-means clustering
│   └── low_rank.py          # Low-rank factorization for convolutional and linear layers
├── pruning/
│   └── importance_pruning.py # Magnitude-based and structured pruning methods
├── distillation/
│   └── knowledge_distillation.py # Knowledge distillation techniques
├── quantization/
│   └── quantization.py      # Quantization techniques (static, dynamic, QAT)
├── evaluation/
│   └── metrics.py           # Evaluation metrics and visualization tools
├── utils/
│   ├── model_utils.py       # Utilities for model operations
│   ├── data_utils.py        # Utilities for dataset loading
│   └── logger.py            # Logging utilities
├── main.py                  # Main script for running optimizations
├── config.py                # Configuration management
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd dnn_optimization
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the main script with default settings:

```bash
python main.py --model resnet18 --dataset cifar10 --optimization all
```

### Specific Optimization Techniques

Apply a specific optimization technique:

```bash
# Weight sharing
python main.py --model resnet18 --dataset cifar10 --optimization weight_sharing --n_clusters 32

# Low-rank factorization
python main.py --model resnet18 --dataset cifar10 --optimization low_rank --rank_percent 0.25

# Pruning
python main.py --model resnet18 --dataset cifar10 --optimization pruning --sparsity 0.5 --pruning_method magnitude

# Knowledge distillation
python main.py --model resnet18 --dataset cifar10 --optimization distillation --teacher_model resnet50 --temperature 4.0 --alpha 0.5

# Quantization
python main.py --model resnet18 --dataset cifar10 --optimization quantization --quantization_method static --bit_width 8
```

### Configuration File

You can also use a configuration file to specify all parameters:

```bash
python main.py --config configs/my_config.json
```

Example configuration file (JSON format):
```json
{
    "general": {
        "seed": 42,
        "device": "cuda",
        "log_dir": "logs",
        "results_dir": "results"
    },
    "dataset": {
        "name": "cifar10",
        "batch_size": 128,
        "num_workers": 4,
        "data_dir": "data"
    },
    "model": {
        "name": "resnet18",
        "pretrained": true
    },
    "optimization": {
        "compression": {
            "enabled": true,
            "method": "weight_sharing",
            "weight_sharing": {
                "n_clusters": 32,
                "layers_to_skip": ["conv1", "fc"]
            }
        },
        "pruning": {
            "enabled": true,
            "method": "magnitude",
            "sparsity": 0.5,
            "layers_to_skip": ["conv1", "fc"]
        },
        "distillation": {
            "enabled": false
        },
        "quantization": {
            "enabled": true,
            "method": "static",
            "bit_width": 8,
            "layers_to_skip": ["conv1", "fc"]
        }
    },
    "evaluation": {
        "batch_size": 128,
        "metrics": ["accuracy", "inference_time", "flops", "size", "parameters"],
        "plot_results": true
    }
}
```

## Implemented Techniques

### Network Compression
- **Weight Sharing**: Reduces the number of unique weights in a model using K-means clustering.
- **Low-Rank Factorization**: Decomposes weight matrices into lower-rank approximations.

### Model Pruning
- **Magnitude Pruning**: Removes weights with the smallest absolute values.
- **Structured Pruning**: Removes entire channels or filters based on their importance.

### Knowledge Distillation
- **Standard Distillation**: Trains a smaller student model to mimic a larger teacher model.
- **Feature-based Distillation**: Aligns intermediate feature representations.

### Quantization
- **Static Quantization**: Pre-computes quantization parameters based on calibration data.
- **Dynamic Quantization**: Computes quantization parameters on-the-fly during inference.
- **Quantization-Aware Training (QAT)**: Simulates quantization during training for better accuracy.

## Evaluation Metrics
- Accuracy
- Inference time
- FLOPs (Floating Point Operations)
- Model size
- Number of parameters

## Results Visualization

The project automatically generates comparison plots and saves them in the results directory:
- Accuracy comparison
- Inference time comparison
- FLOPs comparison
- Model size comparison
- Parameters comparison
- Radar chart for overall comparison

## License

[MIT License](LICENSE)
