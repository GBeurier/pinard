"""
The :mod:`pinard.preprocessing` module includes savitzky golay, baseline,
haar, gaussian, etc.
"""


from _Standard import Baseline, Detrend, Gaussian
from _Standard import baseline, detrend, gaussian
from _Scaler import StandardNormalVariate, RobustNormalVariate, Normalize, SimpleScale
from _Scaler import snv, rnv, norml, scale
from _Nirs import MultiplicativeScatterCorrection, SavitzkyGolay, Haar, Wavelet
from _Nirs import msc, emsc, savgol, haar, wavelet


__all__ = [
    "Baseline", 
    "StandardNormalVariate", ## SAME as STANDARDSCALER
    "RobustNormalVariate", ## SAME as ROBUSTSCALER
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