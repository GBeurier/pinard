
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('__file__'), '..')))

import time
from pinard.presets.ref_models import decon, nicon, customizable_nicon, nicon_classification
from pinard.presets.preprocessings import decon_set, nicon_set
from pinard.data_splitters import KennardStoneSplitter
from pinard.transformations import StandardNormalVariate as SNV, SavitzkyGolay as SG, Gaussian as GS, Derivate as  Dv
from pinard.transformations import Rotate_Translate as RT, Spline_X_Simplification as SXS, Random_X_Operation as RXO
from pinard.transformations import CropTransformer
from pinard.core.runner import ExperimentRunner
from pinard.core.config import Config

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold, ShuffleSplit, GroupKFold, StratifiedShuffleSplit, BaseCrossValidator, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


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

bacon_train = {"action": "train", "training_params": {"epochs": 2000, "batch_size": 500, "patience": 200, "cyclic_lr": True, "base_lr": 1e-6, "max_lr": 1e-3, "step_size": 400}}
bacon_train_short = {"action": "train", "training_params": {"epochs": 10, "batch_size": 500, "patience": 20, "cyclic_lr": True, "base_lr": 1e-6, "max_lr": 1e-3, "step_size": 40}}
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
        "verbose":0
    }
}

full_bacon_finetune = {
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


x_pipeline_full = [
    RobustScaler(),
    {"samples": [None, None,None,None,SXS, RXO]},
    {"split": RepeatedKFold(n_splits=3, n_repeats=1)},
    {"features": [None, GS(2,1), SG, SNV, Dv, [GS, SNV], [GS, GS],[GS, SG],[SG, SNV], [GS, Dv], [SG, Dv]]},
    MinMaxScaler()
]


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
        "verbose":0
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

x_pipeline_PLS = [
    RobustScaler(),
    {"split": RepeatedKFold(n_splits=3, n_repeats=1)},
    {"features": [None, GS(2,1), SG, SNV, Dv, [GS, SNV], [GS, GS],[GS, SG],[SG, SNV], [GS, Dv], [SG, Dv]]},
    MinMaxScaler()
]
            
            
x_pipeline = [
    RobustScaler(), 
    {"split": RepeatedKFold(n_splits=3, n_repeats=1)}, 
    # bacon_set(),
    MinMaxScaler()
]

x_pipelineb = [
    RobustScaler(), 
    {"samples": [RT(6)], "balance": True},
    
    # {"samples": [None, RT]},
    
    {"split": RepeatedKFold(n_splits=3, n_repeats=1)}, 
    
    # {"features": 
        
        # },
    
    MinMaxScaler()
]


seed = 123459456

datasets = "sample_data/mock_data3_classif"
y_pipeline = MinMaxScaler()
# processing only
config1 = Config("sample_data/Malaria2024", x_pipeline_full, y_pipeline, None, None, seed)
## TRAINING
# regression
config2 = Config("sample_data/mock_data2", x_pipeline, y_pipeline, nicon, bacon_train_short, seed)
config3 = Config("sample_data/mock_data3", x_pipeline_PLS, y_pipeline, model_sklearn, None, seed)
# classification
config4 = Config("sample_data/mock_data3_classif", x_pipeline, None, nicon_classification, {"task":"classification", "training_params":{"epochs":10, "patience": 100, "verbose":0}}, seed*2)
config4b = Config("sample_data/Malaria2024", x_pipelineb, None, nicon_classification, {"task":"classification", "training_params":{"epochs":10, "patience": 100, "verbose":0}}, seed*2)
config5 = Config("sample_data/mock_data3_binary", x_pipeline, None, nicon_classification, {"task":"classification", "training_params":{"epochs":5}, "verbose":0}, seed*2)
config6 = Config("sample_data/WhiskyConcentration", x_pipeline, None, RandomForestClassifier, {"task":"classification"}, seed*2)
config7 = Config("sample_data/Malaria2024", x_pipeline, None, RandomForestClassifier, {"task":"classification"}, seed*2)
## FINETUNING
# regression
config8 = Config("sample_data/mock_data3", x_pipeline, y_pipeline, nicon, bacon_finetune, seed)
config9 = Config("sample_data/mock_data3", x_pipeline, y_pipeline, model_sklearn, finetune_pls_experiment, seed)
# classification
config10 = Config("sample_data/Malaria2024", x_pipeline, None, nicon_classification, bacon_finetune_classif, seed*2)
config10b = Config("sample_data/mock_data3_classif", x_pipeline, None, nicon_classification, bacon_finetune_classif, seed*2)
config11 = Config("sample_data/Malaria2024", x_pipelineb, None, RandomForestClassifier, finetune_randomForestclassifier, seed*2)
config11b = Config("sample_data/mock_data3_classif", x_pipeline, None, RandomForestClassifier, finetune_randomForestclassifier, seed*2)


# configs = [config1, config2, config3, config4, config4b, config5, config6, config7, config8, config9, config10, config10b, config11, config11b]
configs = [config10b, config11, config11b]
config_names = ["config1", "config2", "config3", "config4", "config4b", "config5", "config6", "config7", "config8", "config9", "config10", "config10b", "config11", "config11b"]
for i, config in enumerate(configs):
    print("#" * 20)
    print(f"Config {i}: {config_names[i]}")
    print("#" * 20)
    start = time.time()
    runner = ExperimentRunner([config], resume_mode="restart")
    dataset, model_manager = runner.run()
    end = time.time()
    print(f"Time elapsed: {end-start} seconds")