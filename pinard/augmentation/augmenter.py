import abc
import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator


class Augmenter(TransformerMixin, BaseEstimator, metaclass=abc.ABCMeta):
    def __init__(self, count=1):
        self.count = count

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X, y)

    def fit(self, X, y=None):
        return self

    @abc.abstractmethod
    def augment(self, X, y):
        new_X = X.copy()
        new_y = y.copy()
        return new_X, new_y

    def transform(self, X, y):
        self.init_size = len(X)
        for i in range(len(X)):
            for k in range(self.count):
                newX, newy = self.augment(X[i], y[i])
                if isinstance(newX, np.ndarray):
                    X = np.vstack([X, newX])
                    y = np.vstack([y, newy])
        return X, y

    def inverse_transform(self, X, y):
        return X[0 : self.init_size], y[0 : self.init_size]

    def _more_tags(self):
        return {"allow_nan": False}
