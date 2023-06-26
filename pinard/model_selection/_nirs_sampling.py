import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.utils.validation import _num_samples

from ..sklearn._utils import _validate_shuffle_split


def _max_min_distance_split(distance, train_size):
    """
    Sample set split method based on maximum minimum distance, which is the core 
    of the Kennard Stone method.

    Parameters
    ----------
    distance : ndarray
        Semi-positive real symmetric matrix of a certain distance metric.

    train_size : int
        Should be greater than 2.

    Returns
    -------
    tuple
        (List of int, List of int)
        Index of selected spectra as train data, index is zero-based.
        Index of remaining spectra as test data, index is zero-based.

    Example
    -------
    >>> index_train, index_test = max_min_distance_split(distance, train_size)
    >>> print(index_test[0:3])
    [6, 22, 33, 39]

    """

    index_train = np.array([]).astype(np.int32)
    index_test = np.arange(distance.shape[0]).astype(np.int32)

    # First select 2 farthest points
    first_2pts = np.unravel_index(np.argmax(distance), distance.shape)
    index_train = np.append(index_train, first_2pts[0])
    index_train = np.append(index_train, first_2pts[1])
    # Remove the first 2 points from the remaining list
    index_test = np.delete(index_test, np.argwhere(index_test == first_2pts[0]))
    index_test = np.delete(index_test, np.argwhere(index_test == first_2pts[1]))

    for i in range(train_size - 2):
        # Find the maximum minimum distance
        select_distance = distance[index_train, :]
        min_distance = select_distance[:, index_test]
        min_distance = np.min(min_distance, axis=0)
        max_min_distance = np.max(min_distance)

        # Select the first point (in case that several distances are the same, choose 
        # the first one)
        points = np.argwhere(select_distance == max_min_distance)[:, 1]
        for point in points:
            if point in index_train:
                pass
            else:
                index_train = np.append(index_train, point)
                index_test = np.delete(index_test, np.argwhere(index_test == point))
                break

    return (index_train, index_test)


def ks_sampling(data, test_size, *, random_state=None, pca_components=None, metric="euclidean"):
    """
    Samples data using the Kennard Stone method.

    Parameters
    ----------
    size : float/int
        Size of the test set.

    data : DataFrame
        Dataset used to get a train set and a test set.

    pca_components : int/float, default=None
        Value to perform PCA.

    metric : str, default="euclidean"
        The distance metric to use, by default 'euclidean'.
        See scipy.spatial.distance.cdist for more information.

    Returns
    -------
    tuple
        (List of int, List of int)
        Index of selected spectra as train data, index is zero-based.
        Index of remaining spectra as test data, index is zero-based.

    Raises
    ------
    ValueError
        If train sample size is not at least 2.

    Example
    -------
    >>> index_train, index_test = ks_sampling(data, 0.2, None, "euclidean")
    >>> print(index_test[0:4])
    [22, 23, 33, 66]

    References
    ----------
    Kennard, R. W., & Stone, L. A. (1969). Computer aided design of experiments.
    Technometrics, 11(1), 137-148. (https://www.jstor.org/stable/1266770)
    """

    n_samples = _num_samples(data)
    n_train, n_test = _validate_shuffle_split(n_samples, test_size, None)

    if pca_components is not None:
        pca = PCA(pca_components, random_state=random_state)
        data = pca.fit_transform(data)

    if n_train > 2:
        distance = cdist(data, data, metric=metric)
        return _max_min_distance_split(distance, n_train)
    else:
        raise ValueError("Train sample size should be at least 2.")


def spxy_sampling(data, y, test_size, *, random_state=None, pca_components=None, metric="euclidean"):
    """
    Samples data using the SPXY method.

    Parameters
    ----------
    size : float/int
        Size of the test set.

    data : DataFrame
        Features used to get a train set and a test set.

    y : DataFrame
        Labels used to get a train set and a test set.

    pca_components : int/float, default=None
        Value to perform PCA.

    metric : str, default="euclidean"
        The distance metric to use, by default 'euclidean'.
        See scipy.spatial.distance.cdist for more information.

    Returns
    -------
    tuple
        (List of int, List of int)
        Index of selected spectra as train data, index is zero-based.
        Index of remaining spectra as test data, index is zero-based.

    Raises
    ------
    ValueError
        If train sample size is not at least 2.
        If y data is not provided.

    Example
    -------
    >>> index_train, index_test = spxy_sampling(data, y, 0.2, None, "euclidean")
    >>> print(index_test[0:4])
    [6, 22, 33, 39]

    References
    ----------
    Galvao et al. (2005). A method for calibration and validation subset partitioning.
    Talanta, 67(4), 736-740. (https://www.sciencedirect.com/science/article/pii/S003991400500192X)
    Li, Wenze, et al. "HSPXY: A hybrid‐correlation and diversity‐distances based data partition 
    method." Journal of Chemometrics 33.4 (2019): e3109.
    """

    if y is None:
        raise ValueError("Y data is required to use SPXY sampling")

    n_samples = _num_samples(data)
    n_train, n_test = _validate_shuffle_split(n_samples, test_size, None)

    if pca_components is not None:
        pca = PCA(pca_components, random_state=random_state)
        data = pca.fit_transform(data)

    # Create Samples
    if n_train > 2:
        distance_features = cdist(data, data, metric=metric)
        distance_features = distance_features / distance_features.max()
        distance_labels = cdist(y, y, metric=metric)
        distance_labels = distance_labels / distance_labels.max()
        distance = distance_features + distance_labels
        return _max_min_distance_split(distance, n_train)
    else:
        raise ValueError("Train sample size should be at least 2.")
