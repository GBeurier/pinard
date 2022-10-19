"""
The :mod:`pinard.preprocessing` module includes savitzky golay, baseline, haar, 
gaussian, etc. TransformerMixins to preprocess NIR spectra.
"""

from ._standard import Baseline, Detrend, Gaussian
from ._standard import baseline, detrend
from ._scaler import (
    StandardNormalVariate,
    RobustNormalVariate,
    Normalize,
    SimpleScale,
    IdentityTransformer,
    Derivate,
)
from ._scaler import norml, derivate, spl_norml
from ._nirs import MultiplicativeScatterCorrection, SavitzkyGolay, Haar, Wavelet
from ._nirs import msc, savgol, wavelet_transform

__all__ = [
    "IdentityTransformer",  ## sklearn.FunctionTransformer alias
    "Baseline",
    "StandardNormalVariate",  ## sklearn.StandardScaler alias
    "RobustNormalVariate",  ## sklearn.ROBUSTSCALER alias
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
]
