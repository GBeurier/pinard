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
    if y is None:
        raise ValueError(
            "Y data are required to use Kbins discretized Stratified sampling"
        )

    discretizer = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
    y_discrete = discretizer.fit_transform(y)

    split_model = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    return next(split_model.split(data, y_discrete))


# TODO Refactor - uniformize varnames, clean code
def kmean_sampling(
    data, test_size, *, random_state=None, pca_components=None, metric="euclidean"
):
    """_summary_

    Parameters
    ----------
    data : _type_
        _description_
    test_size : _type_
        _description_
    random_state : _type_, optional
        _description_, by default None
    pca_components : _type_, optional
        _description_, by default None
    metric : str, optional
        The distance metric to use. If a string, the distance function can be 
        ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’,
        ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulczynski1’, 
        ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, 
        ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’.

    Returns
    -------
    _type_
        _description_
    """
    n_samples = _num_samples(data)
    n_train, n_test = _validate_shuffle_split(n_samples, test_size, None)

    if pca_components is not None:
        pca = PCA(pca_components, random_state=random_state)
        data = pca.fit_transform(data)

    kmean = KMeans(n_train, random_state=random_state)
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
    # """
    # FONCTION d'échantillonnage : ``split_sampling``
    # --------
    # Permet le tirage non-aléatoire d'échantillons dans un dataset, selon la 
    # méthode des supports points.
    # Paramètres
    # ----------
    # * ``size`` : Int/Float
    #         Pourcentage d'échantillons à prélever.
    # * ``data`` : DataFrame
    #         Dataset dans lequel on prélève les échantillons.
    # * ``random_state`` : Int, default=None
    #         Valeur de la seed, pour la reproductibilité des résultats.
    # Return
    # ------
    # * ``Tuple`` : Tuple(List[Int], List[Int])
    #         Retourne un split entre le train_set et le test_set.
    # Exemple
    # -------
    # >>> index_train, index_test = split_sampling(0.2, data, None)
    # >>> print(sorted(index_test))
    # [1, 5, ..., 49, 55, ..., 100, 105]
    # """

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
