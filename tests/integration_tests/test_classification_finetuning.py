"""
Integration tests for classification model finetuning using Pinard API.
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

# Training and finetuning parameters
bacon_finetune_classif = {
    "action": "finetune",
    "task": "classification",
    "finetune_params": {
        "n_trials": 5,
        "model_params": {
            "filters_1": [8, 16, 32, 64], 
            "filters_2": [8, 16, 32, 64], 
            "filters_3": [8, 16, 32, 64]
        }
    },
    "training_params": {
        "epochs": 5,
        "verbose": 0
    }
}

finetune_randomForestclassifier = {
    "action": "finetune",
    "task": "classification",
    "finetune_params": {
        'model_params': {
            'n_estimators': ('int', 5, 20),
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

x_pipelineb = [
    RobustScaler(), 
    {"samples": [RT(6)], "balance": True},
    {"split": RepeatedKFold(n_splits=3, n_repeats=1)}, 
    MinMaxScaler()
]

seed = 123459456
y_pipeline = MinMaxScaler()

@pytest.mark.finetune
@pytest.mark.classification
def test_nicon_classification_finetuning():
    """Test finetuning a Nicon model for classification on mock data."""
    config = Config("sample_data/mock_data3_classif", x_pipeline, None, nicon_classification, bacon_finetune_classif, seed*2)
    
    start = time.time()
    runner = ExperimentRunner([config], resume_mode="restart")
    dataset, model_manager = runner.run()
    end = time.time()
    
    assert dataset is not None, "Dataset should not be None"
    assert model_manager is not None, "Model manager should not be None"
    assert hasattr(model_manager, "best_params"), "Model manager should have best_params attribute"
    print(f"Time elapsed: {end-start} seconds")


@pytest.mark.finetune
@pytest.mark.sklearn
@pytest.mark.classification
def test_random_forest_classification_finetuning():
    """Test finetuning a RandomForest classifier on mock classification data."""
    config = Config("sample_data/mock_data3_classif", x_pipeline, None, RandomForestClassifier, finetune_randomForestclassifier, seed*2)
    
    start = time.time()
    runner = ExperimentRunner([config], resume_mode="restart")
    dataset, model_manager = runner.run()
    end = time.time()
    
    assert dataset is not None, "Dataset should not be None"
    assert model_manager is not None, "Model manager should not be None"
    assert hasattr(model_manager, "best_params"), "Model manager should have best_params attribute"
    print(f"Time elapsed: {end-start} seconds")


@pytest.mark.finetune
@pytest.mark.classification
def test_nicon_classification_finetuning_malaria():
    """Test finetuning a Nicon model for classification on Malaria data."""
    config = Config("sample_data/Malaria2024", x_pipeline, None, nicon_classification, bacon_finetune_classif, seed*2)
    
    start = time.time()
    runner = ExperimentRunner([config], resume_mode="restart")
    dataset, model_manager = runner.run()
    end = time.time()
    
    assert dataset is not None, "Dataset should not be None"
    assert model_manager is not None, "Model manager should not be None"
    assert hasattr(model_manager, "best_params"), "Model manager should have best_params attribute"
    print(f"Time elapsed: {end-start} seconds")


@pytest.mark.finetune
@pytest.mark.sklearn
@pytest.mark.classification
def test_random_forest_classification_finetuning_malaria():
    """Test finetuning a RandomForest classifier on Malaria data."""
    config = Config("sample_data/Malaria2024", x_pipelineb, None, RandomForestClassifier, finetune_randomForestclassifier, seed*2)
    
    start = time.time()
    runner = ExperimentRunner([config], resume_mode="restart")
    dataset, model_manager = runner.run()
    end = time.time()
    
    assert dataset is not None, "Dataset should not be None"
    assert model_manager is not None, "Model manager should not be None"
    assert hasattr(model_manager, "best_params"), "Model manager should have best_params attribute"
    print(f"Time elapsed: {end-start} seconds")


@pytest.mark.finetune
@pytest.mark.classification
def test_multiple_classification_finetuning():
    """Test running multiple classification finetuning experiments."""
    configs = [
        Config("sample_data/mock_data3_classif", x_pipeline, None, nicon_classification, bacon_finetune_classif, seed*2),
        Config("sample_data/Malaria2024", x_pipelineb, None, RandomForestClassifier, finetune_randomForestclassifier, seed*2),
        Config("sample_data/mock_data3_classif", x_pipeline, None, RandomForestClassifier, finetune_randomForestclassifier, seed*2)
    ]
    
    start = time.time()
    runner = ExperimentRunner(configs, resume_mode="restart")
    dataset, model_manager = runner.run()
    end = time.time()
    
    assert dataset is not None, "Dataset should not be None"
    assert model_manager is not None, "Model manager should not be None"
    print(f"Time elapsed: {end-start} seconds")