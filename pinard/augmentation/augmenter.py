import abc
import random

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import FLOAT_DTYPES


class Augmenter(TransformerMixin, BaseEstimator, metaclass=abc.ABCMeta):
    def __init__(self, random_state=None, per_sample=True, *, copy=True):
        self.copy = copy
        self.per_sample = True
        self.random_state = random_state

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X, y)

    def fit(self, X, y=None):
        return self

    @abc.abstractmethod
    def augment(self, X, per_sample=True):
        pass

    def transform(self, X):
        X = self._validate_data(
            X, reset=False, copy=self.copy, dtype=FLOAT_DTYPES, estimator=self
        )
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)
        return self.augment(X, self.per_sample)

    def _more_tags(self):
        return {"allow_nan": False}


class IdentityAugmenter(Augmenter):
    def augment(self, X, _):
        return X
