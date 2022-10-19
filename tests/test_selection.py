import random
import numpy as np
import pytest

from pinard.model_selection import train_test_split_idx

np.random.seed(42)
random.seed(42)

split_list = [
    ({"method": "random", "test_size": 0.25, "random_state": 42}, 0),
    (
        {
            "method": "k_mean",
            "test_size": 0.25,
            "random_state": 42,
            "metric": "canberra",
        },
        1,
    ),
    (
        {
            "method": "k_mean",
            "test_size": 0.25,
            "random_state": 42,
            "metric": "jensenshannon",
        },
        2,
    ),
    (
        {
            "pca_components": 4,
            "method": "k_mean",
            "test_size": 0.25,
            "random_state": 42,
            "metric": "correlation",
        },
        3,
    ),
    ({"method": "kennard_stone", "test_size": 0.25, "random_state": 42}, 4),
    (
        {
            "method": "kennard_stone",
            "test_size": 0.25,
            "random_state": 42,
            "metric": "correlation",
            "pca_components": 8,
        },
        5,
    ),
    (
        {
            "method": "kennard_stone",
            "test_size": 0.25,
            "random_state": 42,
            "metric": "correlation",
        },
        6,
    ),
    ({"method": "spxy", "test_size": 0.25, "random_state": 42}, 7),
    ({"method": "spxy", "test_size": 0.25, "random_state": 42, "pca_components": 2}, 8),
    (
        {"method": "spxy", "test_size": 0.25, "random_state": 42, "metric": "canberra"},
        9,
    ),
    ({"method": "stratified", "test_size": 0.25, "random_state": 42}, 10),
    ({"method": "stratified", "test_size": 0.25, "random_state": 42, "n_bins": 4}, 11),
    ({"method": "circular", "test_size": 0.25, "random_state": 42}, 12),
]


@pytest.mark.parametrize("opt, index", split_list)
def test_model_selection(split_data, split_validation_data, opt, index):
    x, y = split_data
    train_index, _ = train_test_split_idx(x, y=y, **opt)
    np.testing.assert_array_equal(train_index, split_validation_data[:, index])
