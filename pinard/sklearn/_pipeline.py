import numpy as np
from joblib import Parallel, delayed
from sklearn.pipeline import FeatureUnion, _fit_transform_one, _transform_one
from sklearn.preprocessing import FunctionTransformer

# from sklearn.utils.metaestimators import available_if


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

    def transform(self, X):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, X, None, weight)
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

    def __init__(self, transformer_list, *, n_jobs=None, transformer_weights=None, verbose=False):
        """
        This is the constructor for a class that initializes a list of transformers with optional
        parameters.

        :param transformer_list: A list of tuples where each tuple represents a transformer to be
        applied to the data. The tuple can have either two or three elements. If it has two elements,
        the first element is the transformer object and the second element is the name of the
        transformer. If it has three elements, the first element
        :param n_jobs: The number of CPU cores to use for parallel processing. If set to None, all
        available cores will be used
        :param transformer_weights: A dictionary of weights assigned to each transformer. These weights
        are used to compute the weighted average of the transformed features. If not provided, all
        transformers are assumed to have equal weight
        :param verbose: A boolean parameter that controls whether or not progress messages are printed
        to the console during the fitting process. If set to True, progress messages will be printed. If
        set to False, no progress messages will be printed, defaults to False (optional)
        """
        transformer_origin_list = []
        self.augmentation_count = []
        self.total_count = 0
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
            transformer_weights=transformer_weights,
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

    def transform(self, X, y=None):
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
                delayed(_transform_one)(trans, X, y, weight)
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


# def _transform_one(transformer, X, y, weight, **fit_params):
#     res = transformer.transform(X)
#     # if we have a weight for this transformer, multiply output
#     if weight is None:
#         return res
#     return res * weight


# def _transform_one_xy(transformer, X, y, weight, **fit_params):
#     resX, resY = transformer.transform(X, y)
#     # if we have a weight for this transformer, multiply output
#     if weight is None:
#         return resX, resY
#     return resX * weight, resY


# class Pipeline_XY(Pipeline):
#     def _validate_steps(self):
#         pass
#         # names, estimators = zip(*self.steps)

#         # # validate names
#         # self._validate_names(names)

#         # # validate estimators
#         # transformers = estimators[:-1]
#         # estimator = estimators[-1]

#         # for t in transformers:
#         #     if t is None or t == "passthrough":
#         #         continue
#         #     if not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not hasattr(
#         #         t, "transform"
#         #     ):
#         #         raise TypeError(
#         #             "All intermediate steps should be "
#         #             "transformers and implement fit and transform "
#         #             "or be the string 'passthrough' "
#         #             "'%s' (type %s) doesn't" % (t, type(t))
#         #         )

#         # # We allow last estimator to be None as an identity transformation
#         # if (
#         #     estimator is not None
#         #     and estimator != "passthrough"
#         #     and not hasattr(estimator, "fit")
#         # ):
#         #     raise TypeError(
#         #         "Last step of Pipeline should implement fit "
#         #         "or be the string 'passthrough'. "
#         #         "'%s' (type %s) doesn't" % (estimator, type(estimator))
#         #     )


#     def _can_transform(self):
#         return self._final_estimator == "passthrough" or hasattr(
#             self._final_estimator, "transform"
#         )

#     @available_if(_can_transform)
#     def transform(self, X, y = None):
#         """Transform the data, and apply `transform` with the final estimator.

#         Call `transform` of each transformer in the pipeline. The transformed
#         data are finally passed to the final estimator that calls
#         `transform` method. Only valid if the final estimator
#         implements `transform`.

#         This also works where final estimator is `None` in which case all prior
#         transformations are applied.

#         Parameters
#         ----------
#         X : iterable
#             Data to transform. Must fulfill input requirements of first step
#             of the pipeline.

#         Returns
#         -------
#         Xt : ndarray of shape (n_samples, n_transformed_features)
#             Transformed data.
#         """
#         Xt = X
#         yt = y
#         for _, _, transform in self._iter():
#             Xt, yt = transform.transform(Xt, yt)
#         return Xt, yt

#     def _can_inverse_transform(self):
#         return all(hasattr(t, "inverse_transform") for _, _, t in self._iter())

#     @available_if(_can_inverse_transform)
#     def inverse_transform(self, Xt, yt = None):
#         """Apply `inverse_transform` for each step in a reverse order.

#         All estimators in the pipeline must support `inverse_transform`.

#         Parameters
#         ----------
#         Xt : array-like of shape (n_samples, n_transformed_features)
#             Data samples, where ``n_samples`` is the number of samples and
#             ``n_features`` is the number of features. Must fulfill
#             input requirements of last step of pipeline's
#             ``inverse_transform`` method.

#         Returns
#         -------
#         Xt : ndarray of shape (n_samples, n_features)
#             Inverse transformed data, that is, data in the original feature
#             space.
#         """
#         reverse_iter = reversed(list(self._iter()))
#         for _, _, transform in reverse_iter:
#             Xt = transform.inverse_transform(Xt, yt)
#         return Xt
