# experiments/test_pls_experiments.py
import os
import sys
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(parent_dir)

from operators.preprocessings import decon_set, nicon_set
from operators.ref_models import decon, nicon
from experiments.config import Config
from experiments.runner import ExperimentRunner
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold, ShuffleSplit, GroupKFold, StratifiedShuffleSplit, BaseCrossValidator, TimeSeriesSplit
from operators.splitter import KennardStoneSplitter
from operators.preparation import CropTransformer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler
from pinard.augmentation import Rotate_Translate as RT, Spline_X_Simplification as SXS, Random_X_Operation as RXO
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pinard.preprocessing import StandardNormalVariate as SNV, SavitzkyGolay as SG, Gaussian as GS
import os
import sys

class AddVal(TransformerMixin, BaseEstimator):
    def __init__(self, val):
        self.val = val

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X + self.val


model_sklearn = {
    "class": "sklearn.cross_decomposition.PLSRegression",
    "model_params": {
        "n_components": 21,
    }
}

datasets = "mock_data2"
seed = 42
x_pipeline = [
    StandardScaler(),
    decon_set()
    # {"samples": [None, SXS(), RXO()]},
    # {"split": RepeatedKFold(n_splits=3, n_repeats=1)},
    # {"features": [None, [GS(), SNV()], SG(), GS()]},
    # MinMaxScaler()
]
y_pipeline = StandardScaler()
config = Config(datasets, x_pipeline, y_pipeline, decon, {"action": "train", "training_params": {"epochs": 400, "batch_size": 500}}, seed)


runner = ExperimentRunner(config, resume_mode="restart")
dataset, model_manager = runner.run()
# print(dataset)
# print(dataset.to_str("union"))
# print(dataset.to_str("tu"))
