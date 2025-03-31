"""
Example of a PyTorch analysis with Pinard.

This example demonstrates:
- Loading data
- Preprocessing with feature transformations
- Building and training a PyTorch model using Pinard's API
- Making predictions and evaluating the model
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

try:
    # Check if PyTorch is installed
    import torch
    print(f"PyTorch version: {torch.__version__}")
    torch_available = True
except ImportError:
    print("PyTorch not available. This example requires PyTorch.")
    torch_available = False

from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import RepeatedKFold
from pinard.preprocessing import StandardNormalVariate as SNV, SavitzkyGolay as SG, Gaussian as GS
from pinard.augmentation import Random_X_Operation as RXO
from pinard.core.config import Config
from pinard.core.runner import ExperimentRunner

if torch_available:
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Define PyTorch model configuration
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
    
    # Define training parameters
    torch_train = {
        "action": "train", 
        "training_params": {
            "epochs": 50, 
            "batch_size": 32, 
            "patience": 20, 
            "cyclic_lr": True, 
            "base_lr": 1e-6, 
            "max_lr": 1e-3, 
            "step_size": 40
        }
    }
    
    # Define preprocessing pipeline
    x_pipeline = [
        RobustScaler(),
        {"samples": [None, RXO()]},
        {"features": [None, GS(), SG(), SNV(), [GS(), SNV()]]},
        MinMaxScaler()
    ]
    
    y_pipeline = MinMaxScaler()
    
    # Create configuration
    config = Config(
        "WhiskyConcentration",  # Dataset name from sample_data folder
        x_pipeline,
        y_pipeline,
        torch_model,
        torch_train,
        seed=42
    )
    
    print("Starting PyTorch model training with Pinard...")
    start = time.time()
    
    # Run the experiment
    runner = ExperimentRunner([config], resume_mode="restart")
    dataset, model_manager = runner.run()
    
    end = time.time()
    print(f"Time elapsed: {end-start} seconds")
    
    # Print model evaluation metrics
    if hasattr(model_manager, "metrics") and model_manager.metrics:
        print("\nModel Evaluation Metrics:")
        for metric_name, metric_value in model_manager.metrics.items():
            print(f"{metric_name}: {metric_value}")
    
    print("\nPyTorch model training complete!")
    
    # Save model if desired
    try:
        output_dir = os.path.join(script_dir, "../results/sample_dataWhiskyConcentration")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save the model using model_manager's save method if available
        if hasattr(model_manager, "save_model"):
            model_path = os.path.join(output_dir, "pytorch_model.pinard")
            print(f"Saving model to {model_path}")
            model_manager.save_model(model_path)
            print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")

else:
    print("\nSkipping PyTorch example execution due to missing dependencies.")
    print("To run this example, install PyTorch:")
    print("pip install torch")