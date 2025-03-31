"""
Integration tests for Nicon model training.
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

# Training parameters
bacon_train = {"action": "train", "training_params": {"epochs": 100, "batch_size": 32, "patience": 20, "cyclic_lr": True, "base_lr": 1e-6, "max_lr": 1e-3, "step_size": 40}}
bacon_train_short = {"action": "train", "training_params": {"epochs": 10, "batch_size": 32, "patience": 20, "cyclic_lr": True, "base_lr": 1e-6, "max_lr": 1e-3, "step_size": 40}}

# Define pipelines
x_pipeline = [
    RobustScaler(), 
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

@pytest.mark.nicon
def test_nicon_model_training():
    """Test training a Nicon model for regression."""
    config = Config("sample_data/mock_data2", x_pipeline, y_pipeline, nicon, bacon_train_short, seed)
    
    start = time.time()
    runner = ExperimentRunner([config], resume_mode="restart")
    dataset, model_manager = runner.run()
    end = time.time()
    
    assert dataset is not None, "Dataset should not be None"
    assert model_manager is not None, "Model manager should not be None"
    assert len(model_manager.models) > 0, "Model manager should have trained models"
    print(f"Time elapsed: {end-start} seconds")


@pytest.mark.nicon
def test_nicon_model_training_with_features():
    """Test training a Nicon model with feature transformations."""
    config = Config("sample_data/mock_data2", x_pipeline_basic, y_pipeline, nicon, bacon_train_short, seed)
    
    start = time.time()
    runner = ExperimentRunner([config], resume_mode="restart")
    dataset, model_manager = runner.run()
    end = time.time()
    
    assert dataset is not None, "Dataset should not be None"
    assert model_manager is not None, "Model manager should not be None"
    assert len(model_manager.models) > 0, "Model manager should have trained models"
    print(f"Time elapsed: {end-start} seconds")