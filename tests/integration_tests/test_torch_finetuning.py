# """
# Integration tests for PyTorch model finetuning using Pinard API.
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

# # Define finetuning configurations
# finetune_reg_params = {
#     "action": "finetune",
#     "finetune_params": {
#         "n_trials": 5,
#         "model_params": {
#             "layers": [
#                 {"type": "Linear", "in_features": "auto", "out_features": [32, 64, 128]},
#                 {"type": "ReLU"},
#                 {"type": "Dropout", "p": (0.1, 0.5)},
#                 {"type": "Linear", "out_features": [16, 32, 64]},
#                 {"type": "ReLU"},
#                 {"type": "Linear", "out_features": 1}
#             ],
#             "optimizer": ["Adam", "SGD"],
#             "learning_rate": (0.0001, 0.01)
#         }
#     },
#     "training_params": {
#         "epochs": 5,
#         "verbose": 0
#     }
# }

# finetune_class_params = {
#     "action": "finetune",
#     "task": "classification",
#     "finetune_params": {
#         "n_trials": 5,
#         "model_params": {
#             "layers": [
#                 {"type": "Linear", "in_features": "auto", "out_features": [32, 64, 128]},
#                 {"type": "ReLU"},
#                 {"type": "Dropout", "p": (0.1, 0.5)},
#                 {"type": "Linear", "out_features": [16, 32, 64]},
#                 {"type": "ReLU"},
#                 {"type": "Linear", "out_features": 2},
#                 {"type": "LogSoftmax", "dim": 1}
#             ],
#             "optimizer": ["Adam", "SGD"],
#             "learning_rate": (0.0001, 0.01)
#         }
#     },
#     "training_params": {
#         "epochs": 5,
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
# @pytest.mark.finetune
# def test_torch_regression_finetuning():
#     """Test finetuning a PyTorch regression model."""
#     try:
#         import torch
        
#         config = Config("sample_data/WhiskyConcentration", x_pipeline, y_pipeline, torch_reg_model, finetune_reg_params, seed)
        
#         start = time.time()
#         runner = ExperimentRunner([config], resume_mode="restart")
#         dataset, model_manager = runner.run()
#         end = time.time()
        
#         assert dataset is not None, "Dataset should not be None"
#         assert model_manager is not None, "Model manager should not be None"
#         assert hasattr(model_manager, "best_params"), "Model manager should have best_params attribute"
#         print(f"Time elapsed: {end-start} seconds")
#     except ImportError:
#         pytest.skip("PyTorch not available")


# @pytest.mark.torch
# @pytest.mark.finetune
# @pytest.mark.classification
# def test_torch_classification_finetuning():
#     """Test finetuning a PyTorch classification model."""
#     try:
#         import torch
        
#         config = Config("sample_data/Malaria2024", x_pipeline, None, torch_class_model, finetune_class_params, seed)
        
#         start = time.time()
#         runner = ExperimentRunner([config], resume_mode="restart")
#         dataset, model_manager = runner.run()
#         end = time.time()
        
#         assert dataset is not None, "Dataset should not be None"
#         assert model_manager is not None, "Model manager should not be None"
#         assert hasattr(model_manager, "best_params"), "Model manager should have best_params attribute"
#         print(f"Time elapsed: {end-start} seconds")
#     except ImportError:
#         pytest.skip("PyTorch not available")