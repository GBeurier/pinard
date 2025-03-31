"""
Integration tests for finetuning a PLS model using Pinard API.
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

from pinard.transformations import StandardNormalVariate as SNV, SavitzkyGolay as SG, Gaussian as GS, Derivate as Dv
from pinard.transformations import Rotate_Translate as RT, Spline_X_Simplification as SXS, Random_X_Operation as RXO
from pinard.core.runner import ExperimentRunner
from pinard.core.config import Config

from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cross_decomposition import PLSRegression

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

finetune_pls_experiment = {
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
x_pipeline_PLS = [
    RobustScaler(),
    {"split": RepeatedKFold(n_splits=3, n_repeats=1)},
    {"features": [None, GS(2,1), SG, SNV, Dv, [GS, SNV], [GS, GS], [GS, SG], [SG, SNV], [GS, Dv], [SG, Dv]]},
    MinMaxScaler()
]

x_pipeline = [
    RobustScaler(), 
    {"split": RepeatedKFold(n_splits=3, n_repeats=1)}, 
    MinMaxScaler()
]

x_pipeline_basic = [
    RobustScaler(),
    MinMaxScaler()
]

seed = 123459456
y_pipeline = MinMaxScaler()

@pytest.mark.finetune
@pytest.mark.sklearn
def test_pls_finetuning():
    """Test finetuning a PLS model."""
    config = Config("sample_data/mock_data3", x_pipeline, y_pipeline, model_sklearn, finetune_pls_experiment, seed)
    
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
def test_pls_finetuning_with_features():
    """Test finetuning a PLS model with feature transformations."""
    config = Config("sample_data/mock_data3", x_pipeline_PLS, y_pipeline, model_sklearn, finetune_pls_experiment, seed)
    
    start = time.time()
    runner = ExperimentRunner([config], resume_mode="restart")
    dataset, model_manager = runner.run()
    end = time.time()
    
    assert dataset is not None, "Dataset should not be None"
    assert model_manager is not None, "Model manager should not be None"
    assert hasattr(model_manager, "best_params"), "Model manager should have best_params attribute"
    print(f"Time elapsed: {end-start} seconds")