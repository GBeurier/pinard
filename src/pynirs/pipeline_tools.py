from sklearn.pipeline import FeatureUnion, _transform_one, _fit_transform_one
from joblib import Parallel, delayed
from scipy import sparse
import numpy as np

class FeatureUnionNewAxis(FeatureUnion):
    axis = 0
    
    def __init__(self, transformer_list, n_jobs=None, transformer_weights=None, axis=0):
        super().__init__(transformer_list, n_jobs = n_jobs, transformer_weights = transformer_weights)
        self.axis = axis
    
    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(trans, X, y, weight,**fit_params)
            for name, trans, weight in self._iter())

        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        Xs = np.swapaxes(Xs, 0, 1)
        Xs = np.swapaxes(Xs, 1, 2)
        return Xs    
    
    def transform(self, X):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, X, None, weight)
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs = np.swapaxes(Xs, 0, 1)
        Xs = np.swapaxes(Xs, 1, 2)
        return Xs