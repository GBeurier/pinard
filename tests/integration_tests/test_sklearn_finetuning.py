"""
Integration tests for scikit-learn model finetuning using Pinard API.
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

from pinard.core.runner import ExperimentRunner
from pinard.core.config import Config
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Define models
sklearn_reg_model = {
    "class": "sklearn.cross_decomposition.PLSRegression",
    "model_params": {
        "n_components": 10,
    }
}

sklearn_class_model = {
    "class": "sklearn.ensemble.RandomForestClassifier",
    "model_params": {
        "n_estimators": 50,
        "max_depth": 10,
    }
}

# Define finetuning configurations
finetune_reg_params = {
    "action": "finetune",
    "finetune_params": {
        'model_params': {
            'n_components': ('int', 5, 20),
        },
        'training_params': {},
        'tuner': 'sklearn'
    }
}

finetune_class_params = {
    "action": "finetune",
    "task": "classification",
    "finetune_params": {
        'model_params': {
            'n_estimators': ('int', 5, 20),
            'max_depth': ('int', 3, 10),
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

# Dataset and seed configuration
seed = 123459456
y_pipeline = MinMaxScaler()


@pytest.mark.sklearn
@pytest.mark.finetune
def test_sklearn_regression_finetuning():
    """Test finetuning a scikit-learn regression model."""
    config = Config("sample_data/WhiskyConcentration", x_pipeline, y_pipeline, sklearn_reg_model, finetune_reg_params, seed)
    
    start = time.time()
    runner = ExperimentRunner([config], resume_mode="restart")
    dataset, model_manager = runner.run()
    end = time.time()
    print(f"Time elapsed: {end-start} seconds")
    
    assert dataset is not None, "Dataset should not be None"
    assert model_manager is not None, "Model manager should not be None"


@pytest.mark.sklearn
@pytest.mark.finetune
@pytest.mark.classification
def test_sklearn_classification_finetuning():
    """Test finetuning a scikit-learn classification model."""
    config = Config("sample_data/Malaria2024", x_pipeline, None, sklearn_class_model, finetune_class_params, seed)
    
    start = time.time()
    runner = ExperimentRunner([config], resume_mode="restart")
    dataset, model_manager = runner.run()
    end = time.time()
    print(f"Time elapsed: {end-start} seconds")
    
    assert dataset is not None, "Dataset should not be None"
    assert model_manager is not None, "Model manager should not be None"