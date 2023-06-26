import importlib

import numpy as np
from sklearn.utils import indexable

from ._data_driven_sampling import kbins_stratified_sampling, kmean_sampling
from ._nirs_sampling import ks_sampling, spxy_sampling
from ._random_sampling import shuffle_sampling, systematic_circular_sampling

tweening = importlib.util.find_spec("tweening")
if tweening is not None:
    from _data_driven_sampling import split_sampling


def train_test_split_idx(
    *x,
    y=None,
    test_size=None,
    method="random",
    random_state=None,
    metric="euclidean",
    pca_components=None,
    n_bins=10,
    train_size=None,
):
    """
    Split the data into training and test sets based on the specified method.

    Parameters
    ----------
    *x : array-like
        Input arrays to be split. Can be one or more arrays.
    y : array-like, optional
        The target variable.
    test_size : float or int, optional
        If float, represents the proportion of the dataset to include in the test split.
        If int, represents the absolute number of samples to include in the test split.
        Defaults to None.
    method : {'random', 'stratified', 'k_mean', 'kennard_stone', 'spxy', 'circular', 'SPlit'}, optional
        The method used for splitting the data. Defaults to 'random'.
    random_state : int, RandomState instance or None, optional
        Determines random number generation for dataset shuffling. Pass an int for reproducible output
        across multiple function calls. Defaults to None.
    metric : str, optional
        The distance metric to use. If a string, the distance function can be one of the following:
        'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine',
        'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulczynski1',
        'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
        'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'. Defaults to 'euclidean'.
    pca_components : int or None, optional
        The number of components to keep in PCA transformation. If None, no PCA transformation is applied.
        Defaults to None.
    n_bins : int, optional
        The number of bins for stratified sampling. Defaults to 10.
    train_size : float or int, optional
        If float, represents the proportion of the dataset to include in the train split.
        If int, represents the absolute number of samples to include in the train split.
        Defaults to None.

    Returns
    -------
    train_index : ndarray
        The indices of the samples in the training set.
    test_index : ndarray
        The indices of the samples in the test set.

    Raises
    ------
    ValueError
        If `method` is not one of the supported methods.
    ModuleNotFoundError
        If the 'tweening' package is not found when using 'SPlit' method.

    Notes
    -----
    The 'SPlit' method requires the 'tweening' package to be installed.
    See https://github.com/GBeurier/pinard for more information.
    """

    data = indexable(*x)[0]

    if method == "random":
        return shuffle_sampling(
            data,
            test_size,
            random_state=random_state,
        )
    elif method == "stratified":
        return kbins_stratified_sampling(
            data,
            y,
            test_size,
            random_state=random_state,
            n_bins=n_bins,
        )
    elif method == "k_mean":
        return kmean_sampling(
            data,
            test_size=test_size,
            random_state=random_state,
            pca_components=pca_components,
            metric=metric,
        )
    elif method == "kennard_stone":
        return ks_sampling(
            data,
            test_size,
            random_state=random_state,
            pca_components=pca_components,
            metric=metric,
        )
    elif method == "spxy":
        return spxy_sampling(
            data,
            y,
            test_size,
            random_state=random_state,
            pca_components=pca_components,
            metric=metric,
        )
    elif method == "circular":
        return systematic_circular_sampling(
            data,
            y,
            test_size,
            random_state,
        )
    elif method == "SPlit":
        if tweening is None:
            raise ModuleNotFoundError(
                "Cannot use SPlit sampling without tweening package. "
                "See https://github.com/GBeurier/pinard"
            )
        return split_sampling(
            data,
            test_size,
            random_state,
        )
    else:
        raise ValueError(
            "Argument 'tech' must be : ['k_mean' ; 'kennard_stone' ; 'random' ; "
            "'SPlit' ; 'spxy' ; 'stratified' ; 'circular']."
        )
