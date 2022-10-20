import abc

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted


class Augmenter(TransformerMixin, BaseEstimator, metaclass=abc.ABCMeta):
    def __init__(self, *, copy=True):
        self.copy = copy

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X, y)

    def fit(self, X, y=None):
        return self

    @abc.abstractmethod
    def augment(self, X):
        pass

    def transform(self, X):
        X = self._validate_data(
            X, reset=False, copy=self.copy, dtype=FLOAT_DTYPES, estimator=self
        )

        return self.augment(X)

    def _more_tags(self):
        return {"allow_nan": False}


class IdentityAugmenter(Augmenter):
    def augment(self, X):
        return X
