"""
Example of a simple TensorFlow/Keras analysis with Pinard.

This example demonstrates:
- Loading data
- Preprocessing with feature transformations
- Building and training a TensorFlow model using Pinard's API
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
    # Check if TensorFlow is installed
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    tensorflow_available = True
except ImportError:
    print("TensorFlow not available. This example requires TensorFlow.")
    tensorflow_available = False

from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import RepeatedKFold
from pinard.preprocessing import StandardNormalVariate as SNV, SavitzkyGolay as SG, Gaussian as GS
from pinard.augmentation import Spline_X_Simplification as SXS
from pinard.core.config import Config
from pinard.core.runner import ExperimentRunner

if tensorflow_available:
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Define TensorFlow model configuration
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
    
    # Define training parameters
    tf_train = {
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
        {"samples": [None, SXS()]},
        {"features": [None, GS(), SG(), SNV(), [GS(), SNV()]]},
        MinMaxScaler()
    ]
    
    y_pipeline = MinMaxScaler()
    
    # Create configuration
    config = Config(
        "WhiskyConcentration",  # Dataset name from sample_data folder
        x_pipeline,
        y_pipeline,
        tf_model,
        tf_train,
        seed=42
    )
    
    print("Starting TensorFlow model training with Pinard...")
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
    
    print("\nTensorFlow model training complete!")
    
    # Save model if desired
    try:
        output_dir = os.path.join(script_dir, "../results/sample_dataWhiskyConcentration")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save the model using model_manager's save method if available
        if hasattr(model_manager, "save_model"):
            model_path = os.path.join(output_dir, "tensorflow_model.pinard")
            print(f"Saving model to {model_path}")
            model_manager.save_model(model_path)
            print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")

else:
    print("\nSkipping TensorFlow example execution due to missing dependencies.")
    print("To run this example, install TensorFlow:")
    print("pip install tensorflow")