"""
Integration tests for Nicon classification using Pinard API.
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

x_pipeline_basic = [
    RobustScaler(),
    {"features": [None, SG(), SNV(), GS()]},
    MinMaxScaler()
]

seed = 123459456
y_pipeline = MinMaxScaler()

@pytest.mark.classification
def test_nicon_classification():
    """Test training a Nicon model for classification."""
    config = Config("sample_data/mock_data3_classif", x_pipeline, None, nicon_classification, 
                   {"task": "classification", "training_params": {"epochs": 10, "patience": 100, "verbose": 0}}, seed*2)
    
    start = time.time()
    runner = ExperimentRunner([config], resume_mode="restart")
    dataset, model_manager = runner.run()
    end = time.time()
    
    assert dataset is not None, "Dataset should not be None"
    assert model_manager is not None, "Model manager should not be None"
    assert len(model_manager.models) > 0, "Model manager should have trained models"
    print(f"Time elapsed: {end-start} seconds")


@pytest.mark.classification
def test_nicon_classification_with_augmentation():
    """Test training a Nicon classifier with data augmentation."""
    config = Config("sample_data/Malaria2024", x_pipelineb, None, nicon_classification, 
                   {"task": "classification", "training_params": {"epochs": 10, "patience": 100, "verbose": 0}}, seed*2)
    
    start = time.time()
    runner = ExperimentRunner([config], resume_mode="restart")
    dataset, model_manager = runner.run()
    end = time.time()
    
    assert dataset is not None, "Dataset should not be None"
    assert model_manager is not None, "Model manager should not be None"
    assert len(model_manager.models) > 0, "Model manager should have trained models"
    print(f"Time elapsed: {end-start} seconds")


@pytest.mark.classification
def test_nicon_binary_classification():
    """Test training a Nicon model for binary classification."""
    config = Config("sample_data/mock_data3_binary", x_pipeline, None, nicon_classification, 
                    {"task": "classification", "training_params": {"epochs": 5}, "verbose": 0}, seed*2)
    
    start = time.time()
    runner = ExperimentRunner([config], resume_mode="restart")
    dataset, model_manager = runner.run()
    end = time.time()
    
    assert dataset is not None, "Dataset should not be None"
    assert model_manager is not None, "Model manager should not be None"
    assert len(model_manager.models) > 0, "Model manager should have trained models"
    print(f"Time elapsed: {end-start} seconds")