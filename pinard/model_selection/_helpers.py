import importlib

from sklearn.utils import indexable

from ._data_driven_sampling import kbins_stratified_sampling, kmean_sampling
from ._nirs_sampling import ks_sampling, spxy_sampling
from ._random_sampling import shuffle_sampling, systematic_circular_sampling

tweening = importlib.util.find_spec("tweening")
if tweening is not None:
    from _data_driven_sampling import split_sampling


# The distance metric to use. If a string, the distance function can be
#         ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’,
#         ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulczynski1’,
#         ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’,
#         ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’.


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
