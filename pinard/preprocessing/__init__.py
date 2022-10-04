"""
The :mod:`pinard.preprocessing` module includes savitzky golay, baseline,
haar, gaussian, etc.
"""

from _standard import Baseline, Detrend, Gaussian
from _standard import baseline, detrend, gaussian
from _scaler import StandardNormalVariate, RobustNormalVariate, Normalize, SimpleScale
from _scaler import snv, rnv, norml, scale
from _nirs import MultiplicativeScatterCorrection, SavitzkyGolay, Haar, Wavelet
from _nirs import msc, emsc, savgol, haar, wavelet

__all__ = [
    "Baseline", 
    "StandardNormalVariate", ## SAME as sklearn.STANDARDSCALER
    "RobustNormalVariate", ## SAME as sklearn.ROBUSTSCALER
    "SavitzkyGolay",
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
    "emsc",
]