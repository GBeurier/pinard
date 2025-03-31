"""
Example of hyperparameter tuning with Pinard.

This example demonstrates:
- Loading data
- Preprocessing with feature transformations
- Hyperparameter tuning for different model types:
  - scikit-learn PLS model
  - TensorFlow model
  - PyTorch model
- Making predictions and evaluating the best models
"""

import os
import sys
import numpy as np
import time

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

parent_dir = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.append(parent_dir)

from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import RepeatedKFold
from pinard.preprocessing import StandardNormalVariate as SNV, SavitzkyGolay as SG, Gaussian as GS
from pinard.core.config import Config
from pinard.core.runner import ExperimentRunner

# Check for available deep learning frameworks
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    tensorflow_available = True
except ImportError:
    print("TensorFlow not available")
    tensorflow_available = False

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    torch_available = True
except ImportError:
    print("PyTorch not available")
    torch_available = False

# Set random seeds for reproducibility
np.random.seed(42)
if tensorflow_available:
    tf.random.set_seed(42)
if torch_available:
    torch.manual_seed(42)

# Define preprocessing pipeline - standard for all models
x_pipeline = [
    RobustScaler(),
    {"features": [None, GS(), SG(), SNV(), [GS(), SNV()]]},
    MinMaxScaler()
]

y_pipeline = MinMaxScaler()
seed = 42

# scikit-learn PLS model with hyperparameter tuning configuration
pls_model = {
    "class": "sklearn.cross_decomposition.PLSRegression",
    "model_params": {
        "n_components": 10,
    }
}

pls_finetune = {
    "action": "finetune",
    "finetune_params": {
        'model_params': {
            'n_components': ('int', 5, 20),
        },
        'training_params': {},
        'tuner': 'sklearn'
    }
}

# TensorFlow model with hyperparameter tuning configuration
tf_model = {
    "class": "pinard.core.model.TensorFlowModel",
    "model_params": {
        "layers": [
            {"type": "Dense", "units": 64, "activation": "relu"},
            {"type": "Dropout", "rate": 0.2},
            {"type": "Dense", "units": 32, "activation": "relu"},
            {"type": "Dropout", "rate": 0.1},
            {"type": "Dense", "units": 16, "activation": "relu"},
            {"type": "Dense", "units": 1, "activation": "linear"}
        ],
        "optimizer": "adam",
        "loss": "mse"
    }
}

tf_finetune = {
    "action": "finetune",
    "n_trials": 10,
    "finetune_params": {
        "model_params": {
            "layers": [
                {"type": "Dense", "units": [32, 64, 128], "activation": ["relu", "selu"]},
                {"type": "Dropout", "rate": (0.1, 0.5)},
                {"type": "Dense", "units": [16, 32, 64], "activation": ["relu", "selu"]},
                {"type": "Dense", "units": 1, "activation": "linear"}
            ],
            "optimizer": ["adam", "rmsprop"],
            "learning_rate": (0.0001, 0.01)
        }
    },
    "training_params": {
        "epochs": 10,
        "batch_size": [16, 32, 64]
    }
}

# PyTorch model with hyperparameter tuning configuration
torch_model = {
    "class": "pinard.core.model.TorchModel",
    "model_params": {
        "layers": [
            {"type": "Linear", "in_features": "auto", "out_features": 64},
            {"type": "ReLU"},
            {"type": "Dropout", "p": 0.2},
            {"type": "Linear", "in_features": 64, "out_features": 32},
            {"type": "ReLU"},
            {"type": "Linear", "in_features": 32, "out_features": 1}
        ],
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "loss": "MSELoss"
    }
}

torch_finetune = {
    "action": "finetune",
    "n_trials": 10,
    "finetune_params": {
        "model_params": {
            "layers": [
                {"type": "Linear", "in_features": "auto", "out_features": [32, 64, 128]},
                {"type": "ReLU"},
                {"type": "Dropout", "p": (0.1, 0.5)},
                {"type": "Linear", "out_features": [16, 32, 64]},
                {"type": "ReLU"},
                {"type": "Linear", "out_features": 1}
            ],
            "optimizer": ["Adam", "SGD"],
            "learning_rate": (0.0001, 0.01)
        }
    },
    "training_params": {
        "epochs": 10,
        "batch_size": [16, 32, 64]
    }
}

# Create configurations for each model type
configs = []

# Always include scikit-learn PLS model
pls_config = Config("WhiskyConcentration", x_pipeline, y_pipeline, pls_model, pls_finetune, seed)
configs.append(pls_config)

# Add TensorFlow model if available
if tensorflow_available:
    tf_config = Config("WhiskyConcentration", x_pipeline, y_pipeline, tf_model, tf_finetune, seed)
    configs.append(tf_config)

# Add PyTorch model if available
if torch_available:
    torch_config = Config("WhiskyConcentration", x_pipeline, y_pipeline, torch_model, torch_finetune, seed)
    configs.append(torch_config)

print(f"Starting hyperparameter tuning for {len(configs)} model types...")
start = time.time()

# Run all hyperparameter tuning experiments
runner = ExperimentRunner(configs, resume_mode="restart")
dataset, model_managers = runner.run()

end = time.time()
print(f"Total time elapsed: {end-start} seconds")

# Print best parameters and metrics for each model
print("\n=== Best Hyperparameters and Performance ===")

for i, model_manager in enumerate(model_managers if isinstance(model_managers, list) else [model_managers]):
    if i == 0:
        model_type = "scikit-learn PLS"
    elif i == 1 and tensorflow_available:
        model_type = "TensorFlow"
    elif (i == 2 and tensorflow_available and torch_available) or (i == 1 and not tensorflow_available and torch_available):
        model_type = "PyTorch"
    else:
        model_type = f"Model {i}"
        
    print(f"\n--- {model_type} Model ---")
    
    if hasattr(model_manager, "best_params"):
        print("Best parameters:")
        for param_name, param_value in model_manager.best_params.items():
            print(f"  {param_name}: {param_value}")
    
    if hasattr(model_manager, "metrics") and model_manager.metrics:
        print("Performance metrics:")
        for metric_name, metric_value in model_manager.metrics.items():
            print(f"  {metric_name}: {metric_value}")

# Save best models if possible
output_base_dir = os.path.join(script_dir, "../results/sample_dataWhiskyConcentration")
if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)

try:
    for i, model_manager in enumerate(model_managers if isinstance(model_managers, list) else [model_managers]):
        if i == 0:
            model_name = "pls"
        elif i == 1 and tensorflow_available:
            model_name = "tensorflow"
        elif (i == 2 and tensorflow_available and torch_available) or (i == 1 and not tensorflow_available and torch_available):
            model_name = "pytorch"
        else:
            model_name = f"model_{i}"
            
        # Save the model using model_manager's save method if available
        if hasattr(model_manager, "save_model"):
            model_path = os.path.join(output_base_dir, f"{model_name}_tuned_model.pinard")
            print(f"Saving {model_name} model to {model_path}")
            model_manager.save_model(model_path)
    
    print("Models saved successfully.")
except Exception as e:
    print(f"Error saving models: {e}")

print("\nHyperparameter tuning complete!")