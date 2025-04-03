# """
# Integration tests for PyTorch models using Pinard API.
# """

# import pytest
# import os
# import sys
# import time
# import warnings
# from sklearn.exceptions import ConvergenceWarning

# try:
#     script_dir = os.path.dirname(os.path.abspath(__file__))
# except NameError:
#     script_dir = os.getcwd()

# parent_dir = os.path.abspath(os.path.join(script_dir, '../..'))
# sys.path.append(parent_dir)

# from pinard.core.runner import ExperimentRunner
# from pinard.core.config import Config
# from sklearn.model_selection import RepeatedKFold
# from sklearn.preprocessing import MinMaxScaler, RobustScaler

# warnings.filterwarnings("ignore", category=ConvergenceWarning)
# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)

# # Define models
# torch_reg_model = {
#     "class": "pinard.core.model.TorchModel",
#     "model_params": {
#         "layers": [
#             {"type": "Linear", "in_features": "auto", "out_features": 64},
#             {"type": "ReLU"},
#             {"type": "Linear", "in_features": 64, "out_features": 32},
#             {"type": "ReLU"},
#             {"type": "Linear", "in_features": 32, "out_features": 1}
#         ]
#     }
# }

# torch_class_model = {
#     "class": "pinard.core.model.TorchModel",
#     "model_params": {
#         "layers": [
#             {"type": "Linear", "in_features": "auto", "out_features": 64},
#             {"type": "ReLU"},
#             {"type": "Linear", "in_features": 64, "out_features": 32},
#             {"type": "ReLU"},
#             {"type": "Linear", "in_features": 32, "out_features": 2},
#             {"type": "LogSoftmax", "dim": 1}
#         ]
#     }
# }

# # Training parameters
# train_params = {
#     "action": "train", 
#     "training_params": {
#         "epochs": 10, 
#         "batch_size": 32, 
#         "patience": 5,
#         "verbose": 0
#     }
# }

# class_train_params = {
#     "action": "train",
#     "task": "classification", 
#     "training_params": {
#         "epochs": 10, 
#         "batch_size": 32, 
#         "patience": 5,
#         "verbose": 0
#     }
# }

# # Define pipelines
# x_pipeline = [
#     RobustScaler(), 
#     {"split": RepeatedKFold(n_splits=3, n_repeats=1)}, 
#     MinMaxScaler()
# ]

# # Dataset and seed configuration
# seed = 123459456
# y_pipeline = MinMaxScaler()


# @pytest.mark.torch
# def test_torch_regression():
#     """Test running a PyTorch regression model."""
#     try:
#         import torch
        
#         config = Config("sample_data/WhiskyConcentration", x_pipeline, y_pipeline, torch_reg_model, train_params, seed)
        
#         start = time.time()
#         runner = ExperimentRunner([config], resume_mode="restart")
#         datasets, predictions, scores, best_params = runner.run()
#         end = time.time()
        
#         # Since we're using a list of configs, get the first dataset
#         dataset = datasets[0]
#         assert dataset is not None, "Dataset should not be None"
#         print(f"Time elapsed: {end-start} seconds")
#     except ImportError:
#         pytest.skip("PyTorch not available")


# @pytest.mark.torch
# @pytest.mark.classification
# def test_torch_classification():
#     """Test running a PyTorch classification model."""
#     try:
#         import torch
        
#         config = Config("sample_data/Malaria2024", x_pipeline, None, torch_class_model, class_train_params, seed)
        
#         start = time.time()
#         runner = ExperimentRunner([config], resume_mode="restart")
#         datasets, predictions, scores, best_params = runner.run()
#         end = time.time()
        
#         # Since we're using a list of configs, get the first dataset
#         dataset = datasets[0]
#         assert dataset is not None, "Dataset should not be None"
#         print(f"Time elapsed: {end-start} seconds")
#     except ImportError:
#         pytest.skip("PyTorch not available")