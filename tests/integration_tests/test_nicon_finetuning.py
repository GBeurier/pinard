"""
Integration tests for Nicon model finetuning using Pinard API.
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

from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Training and finetuning parameters
bacon_train = {"action": "train", "training_params": {"epochs": 100, "batch_size": 32, "patience": 20, "cyclic_lr": True, "base_lr": 1e-6, "max_lr": 1e-3, "step_size": 40}}
bacon_train_short = {"action": "train", "training_params": {"epochs": 10, "batch_size": 32, "patience": 20, "cyclic_lr": True, "base_lr": 1e-6, "max_lr": 1e-3, "step_size": 40}}

bacon_finetune = {
    "action": "finetune",
    "finetune_params": {
        "n_trials": 5,
        "model_params": {
            "filters_1": [8, 16, 32, 64], 
            "filters_2": [8, 16, 32, 64], 
            "filters_3": [8, 16, 32, 64]
        }
    },
    "training_params": {
        "epochs": 10,
        "verbose": 0
    }
}

bacon_finetune_full = {
    "action": "finetune",
    "training_params": {
        "epochs": 100,
        "patience": 20,
    },
    "finetune_params": {
        "nb_trials": 50,
        "model_params": {
            'spatial_dropout': (float, 0.01, 0.5),
            'filters1': [4, 8, 16, 32, 64],
            'kernel_size1': [3, 5, 7, 9, 11],
            'dropout_rate': (float, 0.01, 0.5),
            'filters2': [4, 8, 16, 32, 64],
            'activation2': ['relu', 'selu', 'elu'],
            'normalization_method1': ['BatchNormalization', 'LayerNormalization'],
            'filters3': [4, 8, 16, 32, 64],
            'activation3': ['relu', 'selu', 'elu'],
            'dense_activation': ['relu', 'selu', 'elu'],
        }
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

seed = 123459456
y_pipeline = MinMaxScaler()

@pytest.mark.finetune
def test_nicon_finetuning():
    """Test finetuning a Nicon model for regression."""
    config = Config("sample_data/mock_data3", x_pipeline, y_pipeline, nicon, bacon_finetune, seed)
    
    start = time.time()
    runner = ExperimentRunner([config], resume_mode="restart")
    dataset, model_manager = runner.run()
    end = time.time()
    
    assert dataset is not None, "Dataset should not be None"
    assert model_manager is not None, "Model manager should not be None"
    assert hasattr(model_manager, "best_params"), "Model manager should have best_params attribute"
    print(f"Time elapsed: {end-start} seconds")


@pytest.mark.finetune
def test_nicon_finetuning_with_features():
    """Test finetuning a Nicon model with feature transformations."""
    config = Config("sample_data/mock_data3", x_pipeline_full, y_pipeline, nicon, bacon_finetune, seed)
    
    start = time.time()
    runner = ExperimentRunner([config], resume_mode="restart")
    dataset, model_manager = runner.run()
    end = time.time()
    
    assert dataset is not None, "Dataset should not be None"
    assert model_manager is not None, "Model manager should not be None"
    assert hasattr(model_manager, "best_params"), "Model manager should have best_params attribute"
    print(f"Time elapsed: {end-start} seconds")


@pytest.mark.finetune
def test_nicon_finetuning_extensive():
    """Test extensive finetuning of a Nicon model."""
    config = Config("sample_data/mock_data3", x_pipeline, y_pipeline, nicon, bacon_finetune_full, seed)
    
    start = time.time()
    runner = ExperimentRunner([config], resume_mode="restart")
    dataset, model_manager = runner.run()
    end = time.time()
    
    assert dataset is not None, "Dataset should not be None"
    assert model_manager is not None, "Model manager should not be None"
    assert hasattr(model_manager, "best_params"), "Model manager should have best_params attribute"
    print(f"Time elapsed: {end-start} seconds")