"""
Integration tests for TensorFlow model using Pinard API.
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

# Model definitions
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

tf_model_complex = {
    "class": "pinard.core.model.TensorFlowModel",
    "model_params": {
        "layers": [
            {"type": "Dense", "units": 128, "activation": "relu"},
            {"type": "Dropout", "rate": 0.3},
            {"type": "Dense", "units": 64, "activation": "relu"},
            {"type": "Dropout", "rate": 0.2},
            {"type": "Dense", "units": 32, "activation": "relu"},
            {"type": "Dense", "units": 1, "activation": "linear"}
        ]
    }
}

# Training parameters
bacon_train = {"action": "train", "training_params": {"epochs": 100, "batch_size": 32, "patience": 20, "cyclic_lr": True, "base_lr": 1e-6, "max_lr": 1e-3, "step_size": 40}}
bacon_train_short = {"action": "train", "training_params": {"epochs": 10, "batch_size": 32, "patience": 20, "cyclic_lr": True, "base_lr": 1e-6, "max_lr": 1e-3, "step_size": 40}}

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

@pytest.mark.tensorflow
def test_tensorflow_model_training():
    """Test training a tensorflow model."""
    try:
        import tensorflow as tf
        
        config = Config("sample_data/mock_data3", x_pipeline, y_pipeline, tf_model, bacon_train_short, seed)
        
        start = time.time()
        runner = ExperimentRunner([config], resume_mode="restart")
        dataset, model_manager = runner.run()
        end = time.time()
        
        assert dataset is not None, "Dataset should not be None"
        assert model_manager is not None, "Model manager should not be None"
        assert len(model_manager.models) > 0, "Model manager should have trained models"
        print(f"Time elapsed: {end-start} seconds")
    except (ImportError, ModuleNotFoundError):
        pytest.skip("TensorFlow not available")


@pytest.mark.tensorflow
def test_tensorflow_model_with_features():
    """Test training a tensorflow model with feature transformations."""
    try:
        import tensorflow as tf
        
        config = Config("sample_data/mock_data3", x_pipeline_full, y_pipeline, tf_model_complex, bacon_train_short, seed)
        
        start = time.time()
        runner = ExperimentRunner([config], resume_mode="restart")
        dataset, model_manager = runner.run()
        end = time.time()
        
        assert dataset is not None, "Dataset should not be None"
        assert model_manager is not None, "Model manager should not be None"
        assert len(model_manager.models) > 0, "Model manager should have trained models"
        print(f"Time elapsed: {end-start} seconds")
    except (ImportError, ModuleNotFoundError):
        pytest.skip("TensorFlow not available")