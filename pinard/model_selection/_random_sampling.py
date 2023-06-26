import random as rd

import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.utils.validation import _num_samples

from ..sklearn._utils import _validate_shuffle_split


def shuffle_sampling(data, test_size, *, random_state=None):
    """
    Performs random shuffling of the data and splits it into train and test sets.

    Parameters
    ----------
    data : array-like
        The input data samples.
    test_size : float
        The proportion of the data to be used as the test set.
    random_state : int, default=None
        Seed value for random number generation.

    Returns
    -------
    train_index : ndarray
        The indices of the samples in the train set.
    test_index : ndarray
        The indices of the samples in the test set.
    """

    split_model = ShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    return next(split_model.split(data))


def systematic_circular_sampling(data, y, test_size, random_state):
    """
    Performs non-random sampling based on the systematic circular sampling method.
    The starting point and the number of rotations are randomly determined.

    Parameters
    ----------
    size : int/float
        The number of samples to be selected, can be expressed as either the count or the proportion.
    data : DataFrame
        The DataFrame containing the samples.
    random_state : int, default=None
        Seed value for result reproducibility.

    Returns
    -------
    train_index : ndarray
        The indices of the samples in the train set.
    test_index : ndarray
        The indices of the samples in the test set.

    Example
    -------
    >>> index_test = systematic_circular_sampling(0.2, data, 1)
    >>> print(sorted(index_test))
    [3, 8, ..., 53, 58, ..., 101, 106]
    """

    if y is None:
        raise ValueError("Y data are required to use systematic circular sampling")

    if random_state is not None:
        rd.seed(random_state)

    n_samples = _num_samples(data)
    n_train, n_test = _validate_shuffle_split(n_samples, test_size, None)

    ordered_idx = np.argsort(y[:, 0], axis=0)
    rotated_idx = np.roll(ordered_idx, rd.randint(0, n_samples))

    step = n_samples / n_train
    indices = [round(step * i) for i in range(n_train)]

    index_train = rotated_idx[indices]
    index_test = np.delete(rotated_idx, indices)
    return (index_train, index_test)
