import numpy as np
from joblib import Parallel, delayed
from sklearn.pipeline import FeatureUnion, _fit_transform_one, _transform_one
from sklearn.preprocessing import FunctionTransformer


class FeatureAugmentation(FeatureUnion):
    """Stacks results of multiple transformer objects in a new axis.

    This estimator extends sklearn.pipeline.FeatureUnion and applies a list
    of transformer objects in parallel to the input data. Then it stacks
    the results. This is useful to combine several feature extraction mechanisms
    into a single transformer.
    """

    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(trans, X, y, weight, **fit_params)
            for _, trans, weight in self._iter()
        )

        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))

        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        Xs = np.swapaxes(Xs, 0, 1)
        Xs = np.swapaxes(Xs, 1, 2)
        return Xs

    def transform(self, X, **params):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, X, None, weight, params)
            for _, trans, weight in self._iter()
        )

        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs = np.swapaxes(Xs, 0, 1)
        Xs = np.swapaxes(Xs, 1, 2)
        return Xs


class SampleAugmentation(FeatureUnion):
    """Applies multiple feature extraction mechanisms to the same input data and concatenates the results.

    Inherits from the `FeatureUnion` class of the `sklearn.pipeline` module.

    Parameters
    ----------
    transformer_list : list of (str, transformer) or (int, str, transformer) tuples
        List of transformer tuples to be applied to the data. Each tuple contains either two or three elements.
        If the tuple has two elements, it is a (str, transformer) tuple where the first element is the name of the transformer and the second element is the transformer object.
        If the tuple has three elements, it is an (int, str, transformer) tuple where the first element is the count of augmentations for that transformer, the second element is the name of the transformer and the third element is the transformer object.
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel. `None` means 1 unless in a `joblib.parallel_backend` context. `-1` means using all processors.
    transformer_weights : dict or None, optional (default=None)
        Multiplicative weights for features per transformer. Keys are transformer names, values are weights.
    verbose : bool, optional (default=False)
        If True, the time elapsed while fitting each transformer will be printed as it is completed.

    Attributes
    ----------
    transformer_list : list of (str, transformer) tuples
        List of (name, trans) tuples specifying the transformer objects to be applied to the data.
    """

    def __init__(self, transformer_list, n_jobs=None, transformer_weights=None, verbose=False):
        """
        This is the constructor for a class that initializes a list of transformers with optional
        parameters.

        :param transformer_list: A list of tuples where each tuple represents a transformer to be
        applied to the data. The tuple can have either two or three elements. If it has two elements,
        the first element is the transformer object and the second element is the name of the
        transformer. If it has three elements, the first element
        :param n_jobs: The number of CPU cores to use for parallel processing. If set to None, all
        available cores will be used
        :param verbose: A boolean parameter that controls whether or not progress messages are printed
        to the console during the fitting process. If set to True, progress messages will be printed. If
        set to False, no progress messages will be printed, defaults to False (optional)
        """
        transformer_origin_list = []
        self.augmentation_count = []
        self.total_count = 0
        self.transformer_weights = transformer_weights
        for tpl in transformer_list:
            if len(tpl) == 2:
                transformer_origin_list.append(tpl)
                self.augmentation_count.append(1)
                self.total_count += 1
            elif len(tpl) == 3:
                transformer_origin_list.append((tpl[1], tpl[2]))
                self.augmentation_count.append(tpl[0])
                self.total_count += tpl[0]
        super().__init__(
            transformer_origin_list,
            n_jobs=n_jobs,
            transformer_weights=None,
            verbose=verbose,
        )

    def _iter(self):
        """
        Generate (name, trans, weight) x count tuples excluding None and
        'drop' transformers.
        """
        get_weight = (self.transformer_weights or {}).get

        for i, (name, trans) in enumerate(self.transformer_list):
            for _ in range(self.augmentation_count[i]):
                if trans == "drop":
                    continue
                if trans == "passthrough":
                    trans = FunctionTransformer(feature_names_out="one-to-one")
                yield (name, trans, get_weight(name))

                
    def _transform_one(transformer, X, y, weight, columns=None, params=None):
        """Call transform and apply weight to output.

        Parameters
        ----------
        transformer : estimator
            Estimator to be used for transformation.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data to be transformed.

        y : ndarray of shape (n_samples,)
            Ignored.

        weight : float
            Weight to be applied to the output of the transformation.

        columns : str, array-like of str, int, array-like of int, array-like of bool, slice
            Columns to select before transforming.

        params : dict
            Parameters to be passed to the transformer's ``transform`` method.
        """
        if params is not None and hasattr(params, 'transform'):
            res = transformer.transform(X, **params.transform)
        else:
            res = transformer.transform(X)

        return res


    def transform(self, X, y=None, **transform_params):
        """Transform X separately by each transformer, concatenate results.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.

        Returns
        -------
        X_t : array-like or sparse matrix of \
                shape (n_samples, sum_n_components)
            The `hstack` of results of transformers. `sum_n_components` is the
            sum of `n_components` (output dimension) over transformers.
        """
        Xs = zip(
            *Parallel(n_jobs=self.n_jobs)(
                delayed(SampleAugmentation._transform_one)(trans, X, y, weight, params=transform_params)
                for name, trans, weight in self._iter()
            )
        )

        if not Xs:
            return np.zeros((self.total_count, 0)), np.zeros((self.total_count, 0))
        else:
            Xs = np.array(list(Xs))

        Xs = np.concatenate(Xs, axis=0)
        Ys = np.repeat(y, self.total_count, axis=0)

        return Xs, Ys




# from sklearn.pipeline import Pipeline
# from joblib import Parallel, delayed
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.preprocessing import FunctionTransformer

# class SampleAugmentation(Pipeline):
#     def __init__(self, transformer_list, n_jobs=None, transformer_weights=None, verbose=False):
        
#         self.n_jobs = n_jobs
#         self.transformer_weights = transformer_weights
#         # self.verbose = verbose
#         # self.augmentation_count = [count for count, name, trans in transformer_list]
#         # self.total_count = sum(self.augmentation_count)
        
#         transformer_origin_list = []
#         self.augmentation_count = []
#         self.total_count = 0
#         self.transformer_weights = transformer_weights
#         for tpl in transformer_list:
#             if len(tpl) == 2:
#                 transformer_origin_list.append(tpl)
#                 self.augmentation_count.append(1)
#                 self.total_count += 1
#             elif len(tpl) == 3:
#                 transformer_origin_list.append((tpl[1], tpl[2]))
#                 self.augmentation_count.append(tpl[0])
#                 self.total_count += tpl[0]
                
#         self.transformer_list = transformer_origin_list
#         super().__init__(
#             self.transformer_list,
#             # n_jobs=n_jobs,
#             # transformer_weights=None,
#             verbose=verbose,
#         )

#     def _iter(self):
#         """
#         Generate (name, trans, weight) x count tuples excluding None and
#         'drop' transformers.
#         """
#         get_weight = (self.transformer_weights or {}).get

#         for i, (name, trans) in enumerate(self.transformer_list):
#             for _ in range(self.augmentation_count[i]):
#                 if trans == "drop":
#                     continue
#                 if trans == "passthrough":
#                     trans = FunctionTransformer(feature_names_out="one-to-one")
#                 yield (name, trans, get_weight(name))

#     def _transform_one(transformer, X, y, weight, columns=None, params=None):
#         """Call transform and apply weight to output."""

#         if params is not None and 'transform' in params:
#             res = transformer.transform(X, **params['transform'])
#         else:
#             res = transformer.transform(X)

#         return res

#     def transform(self, X, y=None, **transform_params):
#         """Transform X separately by each transformer, concatenate results."""
#         Xs = zip(
#             *Parallel(n_jobs=self.n_jobs)(
#                 delayed(SampleAugmentation._transform_one)(trans, X, y, weight, params=transform_params)
#                 for name, trans, weight in self._iter()
#             )
#         )

#         if not Xs:
#             return np.zeros((self.total_count, 0)), np.zeros((self.total_count, 0))
#         else:
#             Xs = np.array(list(Xs))

#         Xs = np.concatenate(Xs, axis=0)
#         Ys = np.repeat(y, self.total_count, axis=0)

#         return Xs, Ys

#     # def transform(self, X, y=None, **transform_params):
#     #     """Transform X separately by each transformer, concatenate results."""
#     #     results = Parallel(n_jobs=self.n_jobs)(
#     #         delayed(SampleAugmentation._transform_one)(trans, X, y, weight, params=transform_params)
#     #         for name, trans, weight in self._iter()
#     #     )
        
#     #     Xs, Ys = zip(*results)

#     #     if not Xs:
#     #         return np.zeros((self.total_count, 0)), np.zeros((self.total_count, 0))
#     #     else:
#     #         Xs = np.concatenate(Xs, axis=0)
#     #         Ys = np.concatenate(Ys, axis=0)

#     #     return Xs, Ys
