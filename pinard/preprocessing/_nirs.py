import numpy as np
import pywt
from scipy import signal, sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, scale
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted


# mode: ['zero', 'constant', 'symmetric', 'periodic', 'smooth', 'periodization',
# 'reflect', 'antisymmetric', 'antireflect']
# wavelet: ['haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus', 'mexh', 'morl',
# 'cgau', 'shan', 'fbsp', 'cmor']
def wavelet_transform(spectra, wavelet, mode="periodization"):
    """Computes transform using pywavelet transform.
    Args:
        spectra < numpy.ndarray > : NIRS data matrix.
        wavelet < str > : wavelet family transformation
        mode < str > : signal extension mode
    Returns:
        spectra < numpy.ndarray > : wavelet and resampled spectra.
    """
    _, wt_coeffs = pywt.dwt(spectra, wavelet=wavelet, mode=mode)
    if len(wt_coeffs[0]) != len(spectra[0]):
        return signal.resample(wt_coeffs, len(spectra[0]), axis=1)
    else:
        return wt_coeffs


class Wavelet(TransformerMixin, BaseEstimator):
    """Single level Discrete Wavelet Transform.

    Performs a discrete wavelet transform on `data`, using a `wavelet` function.
    see: https://pywavelets.readthedocs.io

    Parameters
    --  --  --  --  --
    wavelet : Wavelet object or name, default = 'haar'
        Wavelet to use: ['Haar', 'Daubechies', 'Symlets', 'Coiflets', 'Biorthogonal',
        'Reverse biorthogonal', 'Discrete Meyer (FIR Approximation)'...]
        see: https://www.pybytes.com/pywavelets/ref/wavelets.html#wavelet-families

    mode : str, optional, default = 'periodization'
        Signal extension mode.add
        ['zero', 'constant', 'symmetric', 'periodic', 'smooth', 'periodization',
         'reflect', 'antisymmetric', 'antireflect']
        see: https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-modes.html#ref-modes
    """

    def __init__(self, wavelet="haar", mode="periodization", *, copy=True):
        self.copy = copy
        self.wavelet = wavelet
        self.mode = mode

    def _reset(self):
        pass

    def fit(self, X, y=None):
        """Verify the X data compliance with wavelet transform

        Parameters
        --  --  --  --  --
        X: array-like, spectra
            The data to transform.
        y: (None) - Ignored

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if sparse.issparse(X):
            raise ValueError("Wavelets does not support sparse input")
        return self

    def transform(self, X, copy=None):
        if sparse.issparse(X):
            raise ValueError('Sparse matrices not supported!"')

        X = self._validate_data(
            X, reset=False, copy=self.copy, dtype=FLOAT_DTYPES, estimator=self
        )

        return wavelet_transform(X, self.wavelet, mode=self.mode)

    def _more_tags(self):
        return {"allow_nan": False}


class Haar(Wavelet):
    """Shortcut to the Wavelet haar transform"""

    def __init__(self, *, copy=True):
        super().__init__("haar", "periodization", copy=copy)


def savgol(spectra, window_length=11, polyorder=3, deriv=0, delta=1.0):
    """Perform Savitzky–Golay filtering on the data (also calculates derivatives). 
    This function is a wrapper for scipy.signal.savgol_filter.
    Args:
        spectra < numpy.ndarray > : NIRS data matrix.
        filter_win < int > : Size of the filter window in samples (default 11).
        polyorder < int > : Order of the polynomial estimation (default 3).
        deriv < int > : Order of the derivation (default 0).
    Returns:
        spectra < numpy.ndarray > : NIRS data smoothed with Savitzky-Golay filtering
    """
    return signal.savgol_filter(spectra, window_length, polyorder, deriv, delta=delta)


class SavitzkyGolay(TransformerMixin, BaseEstimator):
    def __init__(self, window_length=11, polyorder=3, deriv=0, delta=1.0, *, copy=True):
        self.copy = copy
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
        self.delta = delta

    def _reset(self):
        pass

    def fit(self, X, y=None):
        if sparse.issparse(X):
            raise ValueError("SavitzkyGolay does not support sparse input")
        return self

    def transform(self, X, copy=None):
        if sparse.issparse(X):
            raise ValueError('Sparse matrices not supported!"')

        X = self._validate_data(
            X, reset=False, copy=self.copy, dtype=FLOAT_DTYPES, estimator=self
        )

        return savgol(
            X,
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=self.deriv,
            delta=self.delta,
        )

    def _more_tags(self):
        return {"allow_nan": False}


class MultiplicativeScatterCorrection(TransformerMixin, BaseEstimator):
    def __init__(self, scale=True, *, copy=True):
        self.copy = copy
        self.scale = scale

    def _reset(self):
        if hasattr(self, "scaler_"):
            del self.scaler_
            del self.a_
            del self.b_

    def fit(self, X, y=None):
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        if sparse.issparse(X):
            raise TypeError("Normalization does not support sparse input")

        first_pass = not hasattr(self, "mean_")
        X = self._validate_data(X, reset=first_pass, dtype=FLOAT_DTYPES, estimator=self)

        tmp_x = X
        if self.scale:
            scaler = StandardScaler(with_std=False)
            scaler.fit(X)
            self.scaler_ = scaler
            tmp_x = scaler.transform(X)

        reference = np.mean(tmp_x, axis=1)

        a = np.empty(X.shape[1], dtype=float)
        b = np.empty(X.shape[1], dtype=float)

        for col in range(X.shape[1]):
            a[col], b[col] = np.polyfit(reference, tmp_x[:, col], deg=1)

        self.a_ = a
        self.b_ = b

        return self

    def transform(self, X):
        check_is_fitted(self)

        X = self._validate_data(
            X, reset=False, copy=self.copy, dtype=FLOAT_DTYPES, estimator=self
        )

        if X.shape[1] != len(self.a_) or X.shape[1] != len(self.b_):
            raise ValueError(
                "Transform cannot be applied with provided X. Bad number of columns."
            )

        if self.scale:
            X = self.scaler_.transform(X)

        for col in range(X.shape[1]):
            a = self.a_[col]
            b = self.b_[col]
            X[:, col] = (X[:, col] - b) / a

        return X

    def inverse_transform(self, X):
        check_is_fitted(self)

        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES)

        if X.shape[1] != len(self.a_) or X.shape[1] != len(self.b_):
            raise ValueError(
                "Inverse transform cannot be applied with provided X. "
                "Bad number of columns."
            )

        for col in range(X.shape[1]):
            a = self.a_[col]
            b = self.b_[col]
            X[:, col] = (X[:, col] * a) + b

        if self.scale:
            X = self.scaler_.inverse_transform(X)
        return X

    def _more_tags(self):
        return {"allow_nan": False}


def msc(spectra, scaled=True):
    """Performs multiplicative scatter correction to the mean.
    Args:
        spectra < numpy.ndarray > : NIRS data matrix.
    Returns:
        spectra < numpy.ndarray > : Scatter corrected NIR spectra.
    """
    if scaled:
        spectra = scale(spectra, with_std=False, axis=0)  # StandardScaler / demean

    reference = np.mean(spectra, axis=1)

    for col in range(spectra.shape[1]):
        a, b = np.polyfit(reference, spectra[:, col], deg=1)
        spectra[:, col] = (spectra[:, col] - b) / a

    return spectra
