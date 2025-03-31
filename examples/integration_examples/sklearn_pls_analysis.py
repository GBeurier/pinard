"""
Example of a scikit-learn PLS analysis with Pinard.

This example demonstrates:
- Loading data
- Preprocessing with feature transformations
- Training a PLS regression model using Pinard's API
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

from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pinard.preprocessing import StandardNormalVariate as SNV, SavitzkyGolay as SG, Gaussian as GS
from pinard.core.config import Config
from pinard.core.runner import ExperimentRunner

# Set random seed for reproducibility
np.random.seed(42)

# Define sklearn PLS model
model_sklearn = {
    "class": "sklearn.cross_decomposition.PLSRegression",
    "model_params": {
        "n_components": 10,
    }
}

# Define preprocessing pipeline
x_pipeline = [
    RobustScaler(),
    {"features": [None, GS(), SG(), SNV(), [GS(), SNV()]]},
    MinMaxScaler()
]

y_pipeline = MinMaxScaler()

# Create configuration
config = Config(
    "WhiskyConcentration",  # Dataset name from sample_data folder
    x_pipeline,
    y_pipeline,
    model_sklearn,
    None,  # No training parameters needed for sklearn models
    seed=42
)

print("Starting scikit-learn PLS analysis with Pinard...")
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

print("\nModel training complete!")

# Optional: Make custom predictions with the model
if hasattr(dataset, "X_test") and hasattr(dataset, "y_test") and hasattr(model_manager, "models"):
    # Get the first model (should only be one in this case)
    model = model_manager.models[0] if isinstance(model_manager.models, list) else model_manager.models
    
    # Make predictions
    y_pred = model.predict(dataset.X_test)
    
    # If the predictions are 2D, flatten them
    if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten()
    
    # Calculate additional metrics
    mae = mean_absolute_error(dataset.y_test, y_pred)
    mse = mean_squared_error(dataset.y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(dataset.y_test, y_pred)
    
    print("\nDetailed Test Set Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

# Save model if desired
try:
    output_dir = os.path.join(script_dir, "../results/sample_dataWhiskyConcentration")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the model using model_manager's save method if available
    if hasattr(model_manager, "save_model"):
        model_path = os.path.join(output_dir, "sklearn_pls_model.pinard")
        print(f"\nSaving model to {model_path}")
        model_manager.save_model(model_path)
        print("Model saved successfully.")
except Exception as e:
    print(f"\nError saving model: {e}")

print("\nComplete scikit-learn PLS analysis workflow finished!")