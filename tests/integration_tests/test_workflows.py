"""
Integration tests for complex workflow combinations using Pinard API.
"""

import pytest
import os
import sys
import time
import warnings
from sklearn.exceptions import ConvergenceWarning

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

parent_dir = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.append(parent_dir)

from pinard.presets.ref_models import decon, nicon, customizable_nicon, nicon_classification
from pinard.presets.preprocessings import decon_set, nicon_set
from pinard.data_splitters import KennardStoneSplitter
from pinard.transformations import StandardNormalVariate as SNV, SavitzkyGolay as SG, Gaussian as GS, Derivate as Dv
from pinard.transformations import Rotate_Translate as RT, Spline_X_Simplification as SXS, Random_X_Operation as RXO
from pinard.transformations import CropTransformer
from pinard.core.runner import ExperimentRunner
from pinard.core.config import Config

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Model definitions
model_sklearn = {
    "class": "sklearn.cross_decomposition.PLSRegression",
    "model_params": {
        "n_components": 21,
    }
}

tf_model = {
    "class": "pinard.core.model.TensorFlowModel",
    "model_params": {
        "layers": [
            {"type": "Dense", "units": 64, "activation": "relu"},
            {"type": "Dropout", "rate": 0.2},
            {"type": "Dense", "units": 32, "activation": "relu"},
            {"type": "Dense", "units": 1, "activation": "linear"}
        ]
    }
}

torch_model = {
    "class": "pinard.core.model.TorchModel",
    "model_params": {
        "layers": [
            {"type": "Linear", "in_features": "auto", "out_features": 64},
            {"type": "ReLU"},
            {"type": "Linear", "in_features": 64, "out_features": 32},
            {"type": "ReLU"},
            {"type": "Linear", "in_features": 32, "out_features": 1}
        ]
    }
}

# Training and finetuning parameters
bacon_train = {"action": "train", "training_params": {"epochs": 100, "batch_size": 32, "patience": 20, "cyclic_lr": True, "base_lr": 1e-6, "max_lr": 1e-3, "step_size": 40}}
bacon_train_short = {"action": "train", "training_params": {"epochs": 10, "batch_size": 32, "patience": 20, "cyclic_lr": True, "base_lr": 1e-6, "max_lr": 1e-3, "step_size": 40}}

finetune_params = {
    "action": "finetune",
    "finetune_params": {
        'model_params': {
            'n_components': ('int', 5, 20),
        },
        'training_params': {},
        'tuner': 'sklearn'
    }
}

# Define pipelines
x_pipeline = [
    RobustScaler(), 
    {"split": RepeatedKFold(n_splits=3, n_repeats=1)}, 
    MinMaxScaler()
]

x_pipeline_full = [
    RobustScaler(),
    {"samples": [None, None, None, None, SXS, RXO]},
    {"split": RepeatedKFold(n_splits=3, n_repeats=1)},
    {"features": [None, GS(2,1), SG, SNV, Dv, [GS, SNV], [GS, GS], [GS, SG], [SG, SNV], [GS, Dv], [SG, Dv]]},
    MinMaxScaler()
]

x_pipelineb = [
    RobustScaler(), 
    {"samples": [RT(6)], "balance": True},
    {"split": RepeatedKFold(n_splits=3, n_repeats=1)}, 
    MinMaxScaler()
]

seed = 123459456
y_pipeline = MinMaxScaler()


@pytest.mark.workflow
def test_mixed_model_workflow():
    """Test running multiple model types in a single workflow."""
    try:
        import tensorflow as tf
        import torch
        
        configs = [
            Config("sample_data/mock_data3", x_pipeline, y_pipeline, nicon, bacon_train_short, seed),
            Config("sample_data/mock_data3", x_pipeline, y_pipeline, model_sklearn, None, seed),
            Config("sample_data/mock_data3", x_pipeline, y_pipeline, tf_model, bacon_train_short, seed),
            Config("sample_data/mock_data3", x_pipeline, y_pipeline, torch_model, bacon_train_short, seed)
        ]
        
        start = time.time()
        runner = ExperimentRunner(configs, resume_mode="restart")
        dataset, model_manager = runner.run()
        end = time.time()
        
        assert dataset is not None, "Dataset should not be None"
        assert model_manager is not None, "Model manager should not be None"
        assert len(model_manager.models) > 0, "Model manager should have trained models"
        print(f"Time elapsed: {end-start} seconds")
    except (ImportError, ModuleNotFoundError):
        pytest.skip("TensorFlow or PyTorch not available")


@pytest.mark.workflow
def test_mixed_task_workflow():
    """Test running regression and classification tasks in a single workflow."""
    configs = [
        Config("sample_data/mock_data3", x_pipeline, y_pipeline, nicon, bacon_train_short, seed),
        Config("sample_data/mock_data3_classif", x_pipeline, None, nicon_classification, 
               {"task": "classification", "training_params": {"epochs": 10, "verbose": 0}}, seed*2),
        Config("sample_data/mock_data3", x_pipeline, y_pipeline, model_sklearn, None, seed),
        Config("sample_data/mock_data3_classif", x_pipeline, None, RandomForestClassifier, 
               {"task": "classification"}, seed*2)
    ]
    
    start = time.time()
    runner = ExperimentRunner(configs, resume_mode="restart")
    dataset, model_manager = runner.run()
    end = time.time()
    
    assert dataset is not None, "Dataset should not be None"
    assert model_manager is not None, "Model manager should not be None"
    assert len(model_manager.models) > 0, "Model manager should have trained models"
    print(f"Time elapsed: {end-start} seconds")


@pytest.mark.workflow
def test_mixed_pipeline_workflow():
    """Test running different preprocessing pipelines in a single workflow."""
    configs = [
        Config("sample_data/mock_data3", x_pipeline, y_pipeline, nicon, bacon_train_short, seed),
        Config("sample_data/mock_data3", x_pipeline_full, y_pipeline, nicon, bacon_train_short, seed),
        Config("sample_data/mock_data3", x_pipelineb, y_pipeline, nicon, bacon_train_short, seed)
    ]
    
    start = time.time()
    runner = ExperimentRunner(configs, resume_mode="restart")
    dataset, model_manager = runner.run()
    end = time.time()
    
    assert dataset is not None, "Dataset should not be None"
    assert model_manager is not None, "Model manager should not be None"
    assert len(model_manager.models) > 0, "Model manager should have trained models"
    print(f"Time elapsed: {end-start} seconds")


@pytest.mark.workflow
def test_mixed_dataset_workflow():
    """Test running multiple datasets in a single workflow."""
    configs = [
        Config("sample_data/mock_data3", x_pipeline, y_pipeline, nicon, bacon_train_short, seed),
        Config("sample_data/mock_data3_classif", x_pipeline, None, nicon_classification, 
               {"task": "classification", "training_params": {"epochs": 10, "verbose": 0}}, seed*2),
        Config("sample_data/Malaria2024", x_pipelineb, None, nicon_classification, 
               {"task": "classification", "training_params": {"epochs": 10, "verbose": 0}}, seed*2),
        Config("sample_data/WhiskyConcentration", x_pipeline, None, RandomForestClassifier, 
               {"task": "classification"}, seed*2)
    ]
    
    start = time.time()
    runner = ExperimentRunner(configs, resume_mode="restart")
    dataset, model_manager = runner.run()
    end = time.time()
    
    assert dataset is not None, "Dataset should not be None"
    assert model_manager is not None, "Model manager should not be None"
    assert len(model_manager.models) > 0, "Model manager should have trained models"
    print(f"Time elapsed: {end-start} seconds")


@pytest.mark.workflow
def test_comprehensive_workflow():
    """Test a comprehensive workflow mixing all aspects: models, tasks, pipelines, and datasets."""
    try:
        import tensorflow as tf
        import torch
        
        configs = [
            # Regression tasks
            Config("sample_data/mock_data3", x_pipeline, y_pipeline, nicon, bacon_train_short, seed),
            Config("sample_data/mock_data3", x_pipeline_full, y_pipeline, tf_model, bacon_train_short, seed),
            Config("sample_data/mock_data3", x_pipelineb, y_pipeline, torch_model, bacon_train_short, seed),
            Config("sample_data/mock_data3", x_pipeline, y_pipeline, model_sklearn, None, seed),
            
            # Classification tasks
            Config("sample_data/mock_data3_classif", x_pipeline, None, nicon_classification, 
                   {"task": "classification", "training_params": {"epochs": 10, "verbose": 0}}, seed*2),
            Config("sample_data/Malaria2024", x_pipelineb, None, nicon_classification, 
                   {"task": "classification", "training_params": {"epochs": 10, "verbose": 0}}, seed*2),
            Config("sample_data/WhiskyConcentration", x_pipeline, None, RandomForestClassifier, 
                   {"task": "classification"}, seed*2)
        ]
        
        start = time.time()
        runner = ExperimentRunner(configs, resume_mode="restart")
        dataset, model_manager = runner.run()
        end = time.time()
        
        assert dataset is not None, "Dataset should not be None"
        assert model_manager is not None, "Model manager should not be None"
        assert len(model_manager.models) > 0, "Model manager should have trained models"
        print(f"Time elapsed: {end-start} seconds")
    except (ImportError, ModuleNotFoundError):
        pytest.skip("TensorFlow or PyTorch not available")