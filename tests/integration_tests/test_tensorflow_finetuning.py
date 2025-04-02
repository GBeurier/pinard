"""
Integration tests for TensorFlow model finetuning using Pinard API.
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
from pinard.core.utils import framework
from pinard.presets.ref_models import nicon, nicon_classification
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import MinMaxScaler, RobustScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# @framework('tensorflow')
# def custom_tf_regression(input_shape, params={}):
#     from tensorflow.keras.models import Sequential
#     from tensorflow.keras.layers import Dense, Input, Flatten
#     model = Sequential()
#     model.add(Input(shape=input_shape))
#     model.add(Dense(32, activation="relu"))
#     model.add(Flatten())
#     model.add(Dense(1, activation="linear"))
#     return model

# @framework('tensorflow')
# def custom_tf_classification(input_shape, num_classes=2, params={}):
#     from tensorflow.keras.models import Sequential
#     from tensorflow.keras.layers import Dense, Input, Flatten
#     model = Sequential()
#     model.add(Input(shape=input_shape))
#     model.add(Dense(32, activation="relu"))
#     model.add(Flatten())
#     if num_classes == 2:
#         model.add(Dense(1, activation="sigmoid"))
#     else:
#         model.add(Dense(num_classes, activation="softmax"))
#     return model

# # Define finetuning configurations
# finetune_reg_params = {
#     "action": "finetune",
#     "finetune_params": {
#         "n_trials": 5,
#         "model_params": {
#             "layers": [
#                 {"type": "Dense", "units": [32, 64, 128], "activation": ["relu", "elu"]},
#                 {"type": "Dropout", "rate": (0.1, 0.5)},
#                 {"type": "Dense", "units": [16, 32, 64], "activation": ["relu", "elu"]},
#                 {"type": "Dense", "units": 1, "activation": "linear"}
#             ]
#         }
#     },
#     "training_params": {
#         "epochs": 5,
#         "verbose": 0
#     }
# }

# finetune_class_params = {
#     "action": "finetune",
#     "task": "classification",
#     "finetune_params": {
#         "n_trials": 5,
#         "model_params": {
#             "layers": [
#                 {"type": "Dense", "units": [32, 64, 128], "activation": ["relu", "elu"]},
#                 {"type": "Dropout", "rate": (0.1, 0.5)},
#                 {"type": "Dense", "units": [16, 32, 64], "activation": ["relu", "elu"]},
#                 {"type": "Dense", "units": 2, "activation": "softmax"}
#             ]
#         }
#     },
#     "training_params": {
#         "epochs": 5,
#         "verbose": 0
#     }
# }

# @framework('tensorflow')
# def customizable_tf(input_shape, params={}):
#     model = Sequential()
#     model.add(Input(shape=input_shape))
#     model.add(SpatialDropout1D(params.get('spatial_dropout', 0.08)))
#     model.add(Conv1D(filters=params.get('filters1', 8), kernel_size=params.get('kernel_size1', 15), strides=params.get('strides1', 5), activation=params.get('activation1', "selu")))
#     model.add(Dropout(params.get('dropout_rate', 0.2)))
#     model.add(Conv1D(filters=params.get('filters2', 64), kernel_size=params.get('kernel_size2', 21), strides=params.get('strides2', 3), activation=params.get('activation2', "relu")))
#     model.add(BatchNormalization() if params.get('normalization_method1', "BatchNormalization") == "BatchNormalization" else LayerNormalization())
#     model.add(Conv1D(filters=params.get('filters3', 32), kernel_size=params.get('kernel_size3', 5), strides=params.get('strides3', 3), activation=params.get('activation3', "elu")))
#     model.add(BatchNormalization() if params.get('normalization_method2', "BatchNormalization") == "BatchNormalization" else LayerNormalization())
#     model.add(Flatten())
#     model.add(Dense(params.get('dense_units', 16), activation=params.get('dense_activation', "sigmoid")))
#     model.add(Dense(1, activation="sigmoid"))
#     return model

nicon_finetune = {
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
        "verbose":0
    }
}

nicon_finetune_classif = {
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
        "verbose":0
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


@pytest.mark.tensorflow
@pytest.mark.finetune
def test_tensorflow_regression_finetuning():
    """Test finetuning a TensorFlow regression model."""
    try:
        import tensorflow as tf
        
        config1 = Config("sample_data/WhiskyConcentration", x_pipeline, y_pipeline, nicon, nicon_finetune, seed)
        
        start = time.time()
        runner = ExperimentRunner([config1], resume_mode="restart")
        dataset, model_manager = runner.run()
        end = time.time()
        print(f"Time elapsed: {end-start} seconds")
        
        assert dataset is not None, "Dataset should not be None"
        assert model_manager is not None, "Model manager should not be None"
    except (ImportError, ModuleNotFoundError):
        pytest.skip("TensorFlow not available")


@pytest.mark.tensorflow
@pytest.mark.finetune
@pytest.mark.classification
def test_tensorflow_classification_finetuning():
    """Test finetuning a TensorFlow classification model."""
    try:
        import tensorflow as tf
        
        config = Config("sample_data/Malaria2024", x_pipeline, None, nicon_classification, nicon_finetune_classif, seed)
        
        start = time.time()
        runner = ExperimentRunner([config], resume_mode="restart")
        dataset, model_manager = runner.run()
        end = time.time()
        print(f"Time elapsed: {end-start} seconds")
        
        assert dataset is not None, "Dataset should not be None"
        assert model_manager is not None, "Model manager should not be None"
    except (ImportError, ModuleNotFoundError):
        pytest.skip("TensorFlow not available")