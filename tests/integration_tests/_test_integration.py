# """
# Integration tests for various complete workflows using Pinard API.
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

# from pinard.presets.ref_models import decon, nicon, customizable_nicon, nicon_classification
# from pinard.presets.preprocessings import decon_set, nicon_set
# from pinard.data_splitters import KennardStoneSplitter
# from pinard.transformations import StandardNormalVariate as SNV, SavitzkyGolay as SG, Gaussian as GS, Derivate as Dv
# from pinard.transformations import Rotate_Translate as RT, Spline_X_Simplification as SXS, Random_X_Operation as RXO
# from pinard.transformations import CropTransformer
# from pinard.core.runner import ExperimentRunner
# from pinard.core.config import Config

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import KFold, RepeatedKFold
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# warnings.filterwarnings("ignore", category=ConvergenceWarning)
# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)

# # Training and finetuning parameters
# bacon_train_short = {"action": "train", "training_params": {"epochs": 10, "batch_size": 32, "patience": 20, "cyclic_lr": True, "base_lr": 1e-6, "max_lr": 1e-3, "step_size": 40}}

# finetune_randomForestclassifier = {
#     "action": "finetune",
#     "task": "classification",
#     "finetune_params": {
#         'model_params': {
#             'n_estimators': ('int', 5, 20),
#         },
#         'training_params': {},
#         'tuner': 'sklearn'
#     }
# }

# # Define pipelines
# x_pipeline = [
#     RobustScaler(), 
#     {"split": RepeatedKFold(n_splits=3, n_repeats=1)}, 
#     MinMaxScaler()
# ]

# x_pipeline_full = [
#     RobustScaler(),
#     {"samples": [None, None, None, None, SXS, RXO]},
#     {"split": RepeatedKFold(n_splits=3, n_repeats=1)},
#     {"features": [None, GS(2,1), SG, SNV, Dv, [GS, SNV], [GS, GS], [GS, SG], [SG, SNV], [GS, Dv], [SG, Dv]]},
#     MinMaxScaler()
# ]

# seed = 123459456

# @pytest.mark.integration
# def test_combined_classification_workflows():
#     """Test running multiple classification configs together."""
#     # Create configs for different classification approaches
#     configs = [
#         Config("sample_data/mock_data3_classif", x_pipeline, None, nicon_classification, 
#                {"task": "classification", "training_params": {"epochs": 10, "patience": 100, "verbose": 0}}, seed*2),
#         Config("sample_data/mock_data3_classif", x_pipeline, None, RandomForestClassifier, 
#                {"task": "classification"}, seed*2)
#     ]
    
#     start = time.time()
#     runner = ExperimentRunner(configs, resume_mode="restart")
#     dataset, model_manager = runner.run()
#     end = time.time()
    
#     assert dataset is not None, "Dataset should not be None"
#     assert model_manager is not None, "Model manager should not be None"
#     print(f"Time elapsed: {end-start} seconds")


# @pytest.mark.integration
# def test_combined_finetuning_workflows():
#     """Test running multiple finetuning configs together."""
#     # Create configs for different finetuning approaches
#     configs = [
#         Config("sample_data/mock_data3_classif", x_pipeline, None, nicon_classification, 
#                {"task": "classification", "training_params": {"epochs": 10, "verbose": 0}}, seed*2),
#         Config("sample_data/mock_data3_classif", x_pipeline, None, RandomForestClassifier, 
#                finetune_randomForestclassifier, seed*2)
#     ]
    
#     start = time.time()
#     runner = ExperimentRunner(configs, resume_mode="restart")
#     dataset, model_manager = runner.run()
#     end = time.time()
    
#     assert dataset is not None, "Dataset should not be None"
#     assert model_manager is not None, "Model manager should not be None"
#     print(f"Time elapsed: {end-start} seconds")


# @pytest.mark.integration
# def test_combined_workflows_with_features():
#     """Test running multiple configs with feature transformations."""
#     configs = [
#         Config("sample_data/mock_data3_classif", x_pipeline_full, None, nicon_classification, 
#                {"task": "classification", "training_params": {"epochs": 10, "verbose": 0}}, seed*2),
#         Config("sample_data/mock_data3", x_pipeline_full, None, RandomForestClassifier, 
#                finetune_randomForestclassifier, seed*2)
#     ]
    
#     start = time.time()
#     runner = ExperimentRunner(configs, resume_mode="restart")
#     dataset, model_manager = runner.run()
#     end = time.time()
    
#     assert dataset is not None, "Dataset should not be None"
#     assert model_manager is not None, "Model manager should not be None"
#     print(f"Time elapsed: {end-start} seconds")