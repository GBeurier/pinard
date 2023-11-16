import importlib
import random as rd

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils.validation import _num_features, _num_samples

from ..sklearn._utils import _validate_shuffle_split

tweening = importlib.util.find_spec("tweening")
if tweening is not None:
    from tweening import twin


def kbins_stratified_sampling(
    data,
    y,
    test_size,
    random_state=None,
    n_bins=10,
    strategy="uniform",
    encode="ordinal",
):
    """
    Perform stratified sampling using KBins discretization.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        The input data.
    y : array-like of shape (n_samples,)
        The target variable.
    test_size : float or int
        If float, represents the proportion of the dataset to include in the test split.
        If int, represents the absolute number of samples to include in the test split.
    random_state : int, RandomState instance or None, optional (default=None)
        Controls the random seed used to shuffle the data.
    n_bins : int, optional (default=10)
        The number of bins to use for discretization.
    strategy : {'uniform', 'quantile', 'kmeans'}, optional (default='uniform')
        The strategy used to define the widths of the bins.
    encode : {'ordinal', 'onehot', 'onehot-dense'}, optional (default='ordinal')
        The encoding scheme used to encode the transformed result.

    Returns
    -------
    train_index : ndarray
        The indices of the training samples.
    test_index : ndarray
        The indices of the test samples.
    """
    if y is None:
        raise ValueError("Y data are required to use Kbins discretized Stratified sampling")

    discretizer = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy,
                                   subsample=200000)
    y_discrete = discretizer.fit_transform(y)

    split_model = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    return next(split_model.split(data, y_discrete))


def kmean_sampling(
    data, test_size, *, random_state=None, pca_components=None, metric="euclidean"
):
    """
    Perform sampling using K-means clustering.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        The input data.
    test_size : float or int
        If float, represents the proportion of the dataset to include in the test split.
        If int, represents the absolute number of samples to include in the test split.
    random_state : int, RandomState instance or None, optional (default=None)
        Controls the random seed used to initialize the centroids.
    pca_components : int or None, optional (default=None)
        The number of principal components to use for dimensionality reduction using PCA.
    metric : str, optional (default='euclidean')
        The distance metric to use. Possible values are:
        'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine',
        'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulczynski1',
        'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
        'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'.

    Returns
    -------
    train_index : ndarray
        The indices of the training samples.
    test_index : ndarray
        The indices of the test samples.
    """
    n_samples = _num_samples(data)
    n_train, _ = _validate_shuffle_split(n_samples, test_size, None)

    if pca_components is not None:
        pca = PCA(pca_components, random_state=random_state)
        data = pca.fit_transform(data)

    kmean = KMeans(n_train, random_state=random_state, n_init=10)
    kmean.fit(data)
    centroids = kmean.cluster_centers_

    index_train = np.zeros(n_samples)
    for i, centroid in enumerate(centroids):
        tmp_array = cdist(data, [centroid], metric=metric).tolist()
        index_train[i] = tmp_array.index(min(tmp_array))

    index_train = np.unique(index_train).astype(np.int32)
    index_test = np.delete(np.arange(n_samples), index_train)
    return index_train, index_test


def split_sampling(data, test_size, *, random_state=None):
    """
    FONCTION d'échantillonnage : ``split_sampling``
    --------
    Permet le tirage non-aléatoire d'échantillons dans un dataset, selon la méthode des supports points.

    Paramètres
    ----------
    size : int or float
        Pourcentage d'échantillons à prélever.
    data : DataFrame
        Dataset dans lequel on prélève les échantillons.
    random_state : int, default=None
        Valeur de la seed, pour la reproductibilité des résultats.

    Return
    ------
    Tuple : (List[int], List[int])
        Retourne un split entre le train_set et le test_set.

    Exemple
    -------
    >>> index_train, index_test = split_sampling(0.2, data, None)
    >>> print(sorted(index_test))
    [1, 5, ..., 49, 55, ..., 100, 105]
    """
    n_samples = _num_samples(data)
    n_features = _num_features(data)
    n_train, n_test = _validate_shuffle_split(n_samples, test_size, None)

    if not isinstance(random_state, int) or random_state not in range(n_features):
        rd.seed(random_state)
        random_state = rd.randint(0, n_features - 1)

    n_samples /= n_features
    r = round(1 / n_samples)
    index_test = twin(data, r, u1=random_state)
    index_train = np.delete(np.arange(n_samples), index_test)
    return (index_train, index_test)
