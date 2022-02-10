from sklearn.pipeline import FeatureUnion, Pipeline, _transform_one, _fit_transform_one
from sklearn.utils.metaestimators import available_if
import numpy as np

from joblib import Parallel, delayed


def _transform_one_xy(transformer, X, y, weight, **fit_params):
    resX, resY = transformer.transform(X, y)
    # if we have a weight for this transformer, multiply output
    if weight is None:
        return resX, resY
    return resX * weight, resY


class FeatureAugmentation(FeatureUnion):   
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
        print("TRANSFORM")
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, X, None, weight)
            for name, trans, weight in self._iter())
        
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs = np.swapaxes(Xs, 0, 1)
        Xs = np.swapaxes(Xs, 1, 2)
        print("ahahah", Xs.shape)
        return Xs

class SampleAugmentation(FeatureUnion):
    def transform(self, X, y):
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
        print(self,np.array(X).shape, np.array(y).shape)
        
        Xs, ys = zip(*Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one_xy)(trans, X, y, weight)
            for name, trans, weight in self._iter()
        ))

        # for u in Xs:
        #     print(np.array(u).shape)
        # print(self,np.array(Xs).shape, np.array(ys).shape)
        
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0)), np.zeros((y.shape[0], 0))

        x_stk = np.vstack(Xs)
        y_stk = np.vstack(ys)
        print(self, x_stk.shape, y_stk.shape)
        
        return x_stk, y_stk


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
