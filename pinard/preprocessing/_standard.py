import numpy as np
from scipy import ndimage, signal, sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted


class Baseline(TransformerMixin, BaseEstimator):
    """Removes baseline (mean) from each spectrum.

    Args:
        copy (bool, optional): _description_. Defaults to True.
    """

    def __init__(self, *, copy=True):
        self.copy = copy

    def _reset(self):
        if hasattr(self, "mean_"):
            del self.mean_

    def fit(self, X, y=None):
        """Compute the minimum and maximum to be used for later scaling.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.
        y : None
            Ignored.
        Returns
        -------
        self : object
            Fitted Baseline object.
        """
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        if sparse.issparse(X):
            raise TypeError("Baseline does not support sparse input")

        first_pass = not hasattr(self, "mean_")
        X = self._validate_data(X, reset=first_pass, dtype=FLOAT_DTYPES, estimator=self)

        self.mean_ = np.mean(X, axis=0)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)

        X = self._validate_data(
            X, reset=False, copy=self.copy, dtype=FLOAT_DTYPES, estimator=self
        )

        X = X - self.mean_
        return X

    def inverse_transform(self, X, y=None):
        check_is_fitted(self)

        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES)

        X = X + self.mean_
        return X

    def _more_tags(self):
        return {"allow_nan": False}


def baseline(spectra):
    """Removes baseline (mean) from each spectrum.
    Args:
        spectra < numpy.ndarray > : NIRS data matrix.
    Returns:
        spectra < numpy.ndarray > : Mean-centered NIRS data matrix
    """

    return spectra - np.mean(spectra, axis=0)


def detrend(spectra, bp=0):
    """Perform spectral detrending to remove linear trend from data.
    Args:
        spectra < numpy.ndarray > : NIRS data matrix.
        bp < list > : A sequence of break points. If given, an individual linear fit is
        performed for each part of data
        between two break points. Break points are specified as indices into data.
    Returns:
        spectra < numpy.ndarray > : Detrended NIR spectra
    """
    return signal.detrend(spectra, bp=bp)


class Detrend(TransformerMixin, BaseEstimator):
    def __init__(self, bp=0, *, copy=True):
        self.copy = copy
        self.bp = bp

    def _reset(self):
        pass

    def fit(self, X, y=None):
        if sparse.issparse(X):
            raise ValueError('Sparse matrices not supported!"')
        return self

    def transform(self, X, copy=None):
        if sparse.issparse(X):
            raise ValueError('Sparse matrices not supported!"')
        X = self._validate_data(
            X, reset=False, copy=self.copy, dtype=FLOAT_DTYPES, estimator=self
        )

        X = detrend(X, bp=self.bp)
        return X

    def _more_tags(self):
        return {"allow_nan": False}


def gaussian(spectra, order=2, sigma=1):
    """Computes 1D gaussian filter using scipy.ndimage gaussian 1d filter.
    Args:
        spectra < numpy.ndarray > : NIRS data matrix.
        order < float > : Order of the derivation.
        sigma < int > : Sigma of the gaussian.
    Returns:
        spectra < numpy.ndarray > : Gaussian NIR spectra.
    """
    return ndimage.gaussian_filter1d(spectra, order=order, sigma=sigma)


class Gaussian(TransformerMixin, BaseEstimator):
    def __init__(self, order=2, sigma=1, *, copy=True):
        self.copy = copy
        self.order = order
        self.sigma = sigma

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

        X = gaussian(X, order=self.order, sigma=self.sigma)

        return X

    def _more_tags(self):
        return {"allow_nan": False}
