"""
Example of model loading and prediction with Pinard.

This example demonstrates:
- Loading a previously trained model from file
- Using the model to make predictions on new data
- Evaluating the prediction results
"""

import os
import sys
import numpy as np

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

parent_dir = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.append(parent_dir)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pinard.core.model_loader import ModelLoader
from pinard.core.dataset import Dataset
from pinard.data.csv_loader import CSVLoader

# Paths to the saved models - these would be created from previous examples
MODEL_PATHS = {
    'pls': os.path.join(script_dir, "../results/sample_dataWhiskyConcentration/sklearn_pls_model.pinard"),
    'tensorflow': os.path.join(script_dir, "../results/sample_dataWhiskyConcentration/tensorflow_model.pinard"),
    'pytorch': os.path.join(script_dir, "../results/sample_dataWhiskyConcentration/pytorch_model.pinard"),
    'pls_tuned': os.path.join(script_dir, "../results/sample_dataWhiskyConcentration/pls_tuned_model.pinard"),
    'tensorflow_tuned': os.path.join(script_dir, "../results/sample_dataWhiskyConcentration/tensorflow_tuned_model.pinard"),
    'pytorch_tuned': os.path.join(script_dir, "../results/sample_dataWhiskyConcentration/pytorch_tuned_model.pinard")
}

# Set random seed for reproducibility
np.random.seed(42)

# Function to load dataset from CSV files
def load_data(dataset_name="WhiskyConcentration"):
    """Load a dataset for prediction."""
    try:
        # Try to load data from the sample_data folder
        dataset_dir = os.path.join(parent_dir, "examples", "sample_data", dataset_name)
        
        # Check if directory exists
        if not os.path.exists(dataset_dir):
            print(f"Dataset directory {dataset_dir} not found.")
            # Generate some random data as a fallback
            X = np.random.rand(20, 100)  # 20 samples, 100 features
            y = np.random.rand(20)       # Target values
            print(f"Generated random data: X shape {X.shape}, y shape {y.shape}")
            return X, y
        
        # Use Pinard's data loader
        loader = CSVLoader(dataset_dir)
        data = loader.load()
        
        # Split into X and y
        X = data["X"] if "X" in data else None
        y = data["y"] if "y" in data else None
        
        if X is not None:
            print(f"Loaded data: X shape {X.shape}, y shape {y.shape if y is not None else 'N/A'}")
            return X, y
        else:
            # Fallback to random data
            X = np.random.rand(20, 100)
            y = np.random.rand(20)
            print(f"Generated random data: X shape {X.shape}, y shape {y.shape}")
            return X, y
    
    except Exception as e:
        print(f"Error loading data: {e}")
        # Generate random data as a fallback
        X = np.random.rand(20, 100)
        y = np.random.rand(20)
        print(f"Generated random data: X shape {X.shape}, y shape {y.shape}")
        return X, y

def evaluate_predictions(y_true, y_pred):
    """Evaluate predictions using common metrics."""
    # Ensure y_pred is the right shape
    if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print("\nPrediction Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}

# Main function to load models and make predictions
def main():
    # Load test data
    print("Loading test data...")
    X_test, y_test = load_data("WhiskyConcentration")
    
    # Check which models are available
    available_models = {}
    for model_name, model_path in MODEL_PATHS.items():
        if os.path.exists(model_path):
            available_models[model_name] = model_path
    
    if not available_models:
        print("\nNo trained models found. Please run the training examples first to generate models.")
        return
    
    print(f"\nFound {len(available_models)} trained models:")
    for model_name in available_models:
        print(f"- {model_name}")
    
    # Load and evaluate each available model
    for model_name, model_path in available_models.items():
        print(f"\n--- Loading and evaluating {model_name} model ---")
        
        try:
            # Load the model using Pinard's ModelLoader
            model_loader = ModelLoader()
            model = model_loader.load(model_path)
            
            print(f"Successfully loaded {model_name} model from {model_path}")
            
            # Make predictions
            print("Making predictions...")
            y_pred = model.predict(X_test)
            
            # Evaluate predictions if we have ground truth
            if y_test is not None:
                evaluate_predictions(y_test, y_pred)
            else:
                print("No ground truth data available for evaluation.")
                
        except Exception as e:
            print(f"Error loading or using {model_name} model: {e}")
    
    print("\nModel loading and prediction examples completed.")

if __name__ == "__main__":
    main()