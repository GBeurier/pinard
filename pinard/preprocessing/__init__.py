"""
The :mod:`pinard.preprocessing` module includes savitzky golay, baseline, haar, 
gaussian, etc. TransformerMixins to preprocess NIR spectra.
"""
from sklearn.preprocessing import FunctionTransformer as IdentityTransformer
from sklearn.preprocessing import RobustScaler as RobustNormalVariate
from sklearn.preprocessing import StandardScaler as StandardNormalVariate

from ._nirs import (Haar, MultiplicativeScatterCorrection, SavitzkyGolay, Wavelet, msc, savgol, wavelet_transform)
from ._scaler import (Derivate, Normalize, SimpleScale, derivate, norml, spl_norml)
from ._standard import Baseline, Detrend, Gaussian, baseline, detrend, gaussian

__all__ = [
    "IdentityTransformer",  # sklearn.preprocessing.FunctionTransformer alias
    "Baseline",
    "StandardNormalVariate",  # sklearn.preprocessing.StandardScaler alias
    "RobustNormalVariate",  # sklearn.preprocessing.RobusScaler alias
    "SavitzkyGolay",
    "Haar",
    "Normalize",
    "Detrend",
    "MultiplicativeScatterCorrection",
    "Derivate",
    "Gaussian",
    "Wavelet",
    "SimpleScale",
    "baseline",
    "savgol",
    "norml",
    "detrend",
    "msc",
    "wavelet_transform",
    "derivate",
    "spl_norml",
    "gaussian",
]
