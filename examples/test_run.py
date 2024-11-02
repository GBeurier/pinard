# experiments/test_pls_experiments.py
import os
import sys
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(parent_dir)


from sklearn.ensemble import RandomForestClassifier
from operators.preprocessings import decon_set, nicon_set
from operators.ref_models import decon, nicon, customizable_nicon, nicon_classification
from experiments.config import Config
from experiments.runner import ExperimentRunner
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold, ShuffleSplit, GroupKFold, StratifiedShuffleSplit, BaseCrossValidator, TimeSeriesSplit
from operators.splitter import KennardStoneSplitter
from operators.preparation import CropTransformer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler
from pinard.augmentation import Rotate_Translate as RT, Spline_X_Simplification as SXS, Random_X_Operation as RXO
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from pinard.preprocessing import StandardNormalVariate as SNV, SavitzkyGolay as SG, Gaussian as GS, Derivate as Dv
import time


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

nicon_train = {"action": "train", "training_params": {"epochs": 2000, "batch_size": 500, "patience": 200, "cyclic_lr": True, "base_lr": 1e-6, "max_lr": 1e-3, "step_size": 400}}
nicon_train_short = {"action": "train", "training_params": {"epochs": 10, "batch_size": 500, "patience": 20, "cyclic_lr": True, "base_lr": 1e-6, "max_lr": 1e-3, "step_size": 40}}
nicon_finetune = {
    "action": "finetune",
    "n_trials": 5,
    "finetune_params": {
        "model_params": {
            "filters_1": [8, 16, 32, 64],
            "filters_2": [8, 16, 32, 64],
            "filters_3": [8, 16, 32, 64]
        }
    },
    "training_params": {
        "epochs": 10,
    }
}

full_nicon_finetune = {
    "action": "finetune",
    "training_params": {
        "epochs": 500,
        "patience": 100,
    },
    "finetune_params": {
        "nb_trials": 150,
        "model_params": {
            'spatial_dropout': (float, 0.01, 0.5),
            'filters1': [4, 8, 16, 32, 64, 128, 256],
            'kernel_size1': [3, 5, 7, 9, 11, 13, 15],
            # 'strides1': [1, 2, 3, 4, 5],
            # 'activation1': ['relu', 'selu', 'elu', 'swish'],
            'dropout_rate': (float, 0.01, 0.5),
            'filters2': [4, 8, 16, 32, 64, 128, 256],
            # 'kernel_size2': [3, 5, 7, 9, 11, 13, 15],
            # 'strides2': [1, 2, 3, 4, 5],
            'activation2': ['relu', 'selu', 'elu', 'swish'],
            'normalization_method1': ['BatchNormalization', 'LayerNormalization'],
            'filters3': [4, 8, 16, 32, 64, 128, 256],
            # 'kernel_size3': [3, 5, 7, 9, 11, 13, 15],
            # 'strides3': [1, 2, 3, 4, 5],
            'activation3': ['relu', 'selu', 'elu', 'swish'],
            # 'normalization_method2': ['BatchNormalization', 'LayerNormalization'],
            # 'dense_units': [4, 8, 16, 32, 64, 128, 256],
            'dense_activation': ['relu', 'selu', 'elu', 'swish'],
        },
        # "training_params": {
        #     "batch_size": [32, 64, 128, 256, 512],
        #     "cyclic_lr": [True, False],
        #     "base_lr": (float, 1e-6, 1e-2),
        #     "max_lr": (float, 1e-3, 1e-1),
        #     "step_size": (int, 500, 5000),
        # },
    }
}

x_pipeline_PLS = [
    RobustScaler(),
    # {"samples": [None, SXS, RXO]},
    # {"split": RepeatedKFold(n_splits=3, n_repeats=1)},
    {"features": [None, GS(2, 1), SG, SNV, Dv, [GS, SNV], [GS, GS], [GS, SG], [SG, SNV], [GS, Dv], [SG, Dv]]},
    MinMaxScaler()
]


x_pipeline_full = [
    RobustScaler(),
    {"samples": [None, None, None, None, SXS, RXO]},
    {"split": RepeatedKFold(n_splits=3, n_repeats=1)},
    {"features": [None, GS(2, 1), SG, SNV, Dv, [GS, SNV], [GS, GS], [GS, SG], [SG, SNV], [GS, Dv], [SG, Dv]]},
    MinMaxScaler()
]

x_pipeline_full2 = [
    RobustScaler(),
    {"samples": [None, None, None, None, SXS, RXO]},
    {"split": RepeatedKFold(n_splits=3, n_repeats=1)},
    {"features": [None, GS(2, 1), SG, SNV, Dv, [GS, SNV], [GS, GS], [GS, SG], [SG, SNV], [GS, Dv], [SG, Dv]]},
    MinMaxScaler()
]

x_pipeline = [
    RobustScaler(),
    # {"samples": [None, SXS]},
    # {"split": RepeatedKFold(n_splits=3, n_repeats=1)},
    {"features": [None, [GS(), SNV()], SG(), GS()]},
    # {"features": [None, GS]},
    # {"features": [None, GS, SG, SNV, Dv, [GS, SNV], [GS, GS],[GS, SG],[SG, SNV], [GS, Dv], [SG, Dv]]},
    # {"features": [None, SG, GS, SNV, [SG, SNV], [GS, SNV], [SG, GS]]},
    # nicon_set(),
    MinMaxScaler()
]
nicon_finetune_classif = nicon_finetune.copy()
nicon_finetune_classif["task"] = "classification"

pls_finetune_classif = finetune_pls_experiment.copy()
pls_finetune_classif["task"] = "classification"

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

seed = 123459456

y_pipeline = MinMaxScaler()
# processing only
config2 = Config("mock_data3", x_pipeline_full, y_pipeline, None, None, seed)
# TRAINING
# regression
config1 = Config("mock_data2", x_pipeline, y_pipeline, nicon, nicon_train_short, seed)
config4 = Config("mock_data3", x_pipeline_PLS, y_pipeline, model_sklearn, None, seed)
# classification
config3 = Config("mock_data3_classif", x_pipeline, None, nicon_classification, {"task": "classification", "training_params": {"epochs": 5}}, seed*2)
config5 = Config("mock_data3_classif", x_pipeline, None, RandomForestClassifier, {"task": "classification"}, seed*2)
# FINETUNING
# regression
config6 = Config("mock_data3", x_pipeline, y_pipeline, nicon, nicon_finetune, seed)
config7 = Config("mock_data3", x_pipeline, y_pipeline, model_sklearn, finetune_pls_experiment, seed)
# classification
config8 = Config("mock_data3_classif", x_pipeline, None, nicon_classification, nicon_finetune_classif, seed*2)
config9 = Config("mock_data3_classif", x_pipeline, None, RandomForestClassifier, finetune_randomForestclassifier, seed*2)

# ALL TESTS
configs = [config1, config2, config3, config4, config5, config6, config7, config8, config9]

start = time.time()
runner = ExperimentRunner(configs, resume_mode="restart")
dataset, model_manager = runner.run()
end = time.time()
print(f"Time elapsed: {end-start} seconds")
