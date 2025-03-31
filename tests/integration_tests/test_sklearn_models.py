"""
Integration tests for sklearn models using Pinard API.
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

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.cross_decomposition import PLSRegression
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

rf_model = {
    "class": "sklearn.ensemble.RandomForestRegressor",
    "model_params": {
        "n_estimators": 50,
        "max_depth": 10
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

seed = 123459456
y_pipeline = MinMaxScaler()

@pytest.mark.sklearn
def test_sklearn_pls_analysis():
    """Test a complete sklearn PLS regression workflow."""
    config = Config("sample_data/mock_data3", x_pipeline_PLS, y_pipeline, model_sklearn, None, seed)
    
    start = time.time()
    runner = ExperimentRunner([config], resume_mode="restart")
    dataset, model_manager = runner.run()
    end = time.time()
    
    assert dataset is not None, "Dataset should not be None"
    assert model_manager is not None, "Model manager should not be None"
    assert hasattr(model_manager, "models"), "Model manager should have models attribute"
    print(f"Time elapsed: {end-start} seconds")


@pytest.mark.sklearn
def test_sklearn_rf_analysis():
    """Test a complete sklearn RandomForest regression workflow."""
    config = Config("sample_data/mock_data3", x_pipeline, y_pipeline, rf_model, None, seed)
    
    start = time.time()
    runner = ExperimentRunner([config], resume_mode="restart")
    dataset, model_manager = runner.run()
    end = time.time()
    
    assert dataset is not None, "Dataset should not be None"
    assert model_manager is not None, "Model manager should not be None"
    assert hasattr(model_manager, "models"), "Model manager should have models attribute"
    print(f"Time elapsed: {end-start} seconds")


@pytest.mark.sklearn
@pytest.mark.classification
def test_random_forest_whisky():
    """Test training a RandomForest classifier on Whisky concentration data."""
    config = Config("sample_data/WhiskyConcentration", x_pipeline, None, RandomForestClassifier, 
                    {"task": "classification"}, seed*2)
    
    start = time.time()
    runner = ExperimentRunner([config], resume_mode="restart")
    dataset, model_manager = runner.run()
    end = time.time()
    
    assert dataset is not None, "Dataset should not be None"
    assert model_manager is not None, "Model manager should not be None"
    assert len(model_manager.models) > 0, "Model manager should have trained models"
    print(f"Time elapsed: {end-start} seconds")


@pytest.mark.sklearn
@pytest.mark.classification
def test_random_forest_malaria():
    """Test training a RandomForest classifier on Malaria data."""
    config = Config("sample_data/Malaria2024", x_pipeline, None, RandomForestClassifier, 
                    {"task": "classification"}, seed*2)
    
    start = time.time()
    runner = ExperimentRunner([config], resume_mode="restart")
    dataset, model_manager = runner.run()
    end = time.time()
    
    assert dataset is not None, "Dataset should not be None"
    assert model_manager is not None, "Model manager should not be None"
    assert len(model_manager.models) > 0, "Model manager should have trained models"
    print(f"Time elapsed: {end-start} seconds")