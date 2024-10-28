# tests/test_dataset.py

import pytest
import numpy as np
from pinard.data.dataset import Dataset


def test_dataset_initialization():
    dataset = Dataset()
    assert dataset.x_train is None
    assert dataset.y_train is None
    assert dataset.x_test is None
    assert dataset.y_test is None


def test_dataset_x_train_setter_getter():
    dataset = Dataset()
    x_train = np.random.rand(100, 10)
    dataset.x_train = x_train
    assert dataset.x_train.shape == (1, 100, 1, 10)
    assert dataset._x_train.shape == (1, 100, 1, 10)


def test_dataset_x_train_invalid_shape():
    dataset = Dataset()
    x_train = np.random.rand(100)
    with pytest.raises(ValueError):
        dataset.x_train = x_train


def test_dataset_y_train_setter_getter():
    dataset = Dataset()
    x_train = np.random.rand(1, 100, 1, 10)
    dataset._x_train = x_train
    y_train = np.random.rand(100)
    dataset.y_train = y_train
    assert dataset.y_train.shape == (100, 1)  # Adjusted expected shape
    assert dataset._y_train.shape == (100, 1)


def test_dataset_y_train_invalid_shape():
    dataset = Dataset()
    x_train = np.random.rand(1, 100, 1, 10)
    dataset._x_train = x_train
    y_train = np.random.rand(99)
    with pytest.raises(ValueError):
        dataset.y_train = y_train


def test_dataset_filter_x():
    dataset = Dataset()
    x_train = np.random.rand(2, 50, 1, 10)
    dataset._x_train = x_train
    filtered_x = dataset.filter_x(x_train, union_type='concat', indices=[0, 1, 2])
    assert filtered_x.shape == (6, 10)


def test_dataset_filter_y():
    dataset = Dataset()
    x_train = np.random.rand(2, 50, 1, 10)
    y_train = np.random.rand(50, 1)
    dataset._x_train = x_train
    dataset._y_train = y_train
    filtered_y = dataset.filter_y(y_train, indices=[0, 1, 2])
    assert filtered_y.shape == (6, 1)


def test_dataset_fold_data_no_folds():
    dataset = Dataset()
    x_train = np.random.rand(2, 50, 1, 10)
    y_train = np.random.rand(50, 1)
    x_test = np.random.rand(2, 20, 1, 10)
    y_test = np.random.rand(20, 1)
    dataset._x_train = x_train
    dataset._y_train = y_train
    dataset._x_test = x_test
    dataset._y_test = y_test
    folds = list(dataset.fold_data())
    assert len(folds) == 1
    x_train_fold, y_train_fold, x_test_fold, y_test_fold = folds[0]
    assert x_train_fold.shape == (100, 10)
    assert y_train_fold.shape == (100, 1)
    assert x_test_fold.shape == (40, 10)
    assert y_test_fold.shape == (40, 1)


def test_dataset_fold_data_with_folds():
    dataset = Dataset()
    x_train = np.random.rand(2, 100, 1, 10)
    y_train = np.random.rand(100, 1)
    dataset._x_train = x_train
    dataset._y_train = y_train
    dataset._folds = [(np.arange(80), np.arange(80, 100))]
    folds = list(dataset.fold_data())
    assert len(folds) == 1
    x_train_fold, y_train_fold, x_val_fold, y_val_fold = folds[0]
    assert x_train_fold.shape == (160, 10)
    assert y_train_fold.shape == (160, 1)
    assert x_val_fold.shape == (40, 10)
    assert y_val_fold.shape == (40, 1)


def test_dataset_inverse_transform():
    dataset = Dataset()
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    y_train = np.array([[1], [2], [3]])
    scaler.fit(y_train)
    dataset.y_transformer = scaler
    y_pred = np.array([[0], [0.5], [1]])
    y_inverse = dataset.inverse_transform(y_pred)
    expected = scaler.inverse_transform(y_pred)
    assert np.allclose(y_inverse, expected)


def test_dataset_str():
    dataset = Dataset()
    x_train = np.random.rand(2, 50, 1, 10)
    y_train = np.random.rand(50, 1)
    dataset._x_train = x_train
    dataset._y_train = y_train
    s = str(dataset)
    assert "Dataset(" in s


def test_dataset_test_is_indices():
    dataset = Dataset()
    dataset._x_test = np.array([0, 1, 2])
    assert dataset.test_is_indices


def test_dataset_y_test_when_indices():
    dataset = Dataset()
    x_train = np.random.rand(2, 5, 1, 10)
    y_train = np.random.rand(5, 1)
    dataset._x_train = x_train
    dataset._y_train = y_train
    dataset._x_test = np.array([0, 1])
    y_test = dataset.y_test
    assert y_test.shape == (4, 1)


def test_dataset_y_test_setter_when_indices():
    dataset = Dataset()
    dataset._x_test = np.array([0, 1, 2])
    with pytest.raises(ValueError):
        dataset.y_test = np.array([1, 2, 3])


def test_dataset_x_test_setter():
    dataset = Dataset()
    x_test = np.random.rand(100, 10)
    dataset.x_test = x_test
    assert dataset.x_test.shape == (1, 100, 1, 10)


def test_dataset_x_test_invalid_shape():
    dataset = Dataset()
    x_test = np.random.rand(100, 10, 5)  # 3D array, invalid shape
    with pytest.raises(ValueError):
        dataset.x_test = x_test


def test_dataset_property_group_test():
    dataset = Dataset()
    dataset._x_train = np.random.rand(2, 5, 1, 10)
    dataset._x_test = np.random.rand(2, 5, 1, 10)
    dataset._group_train = np.array([1, 2, 3, 4, 5])
    dataset._group_test = np.array([1, 2, 3, 4, 5])
    group_test = dataset.group_test
    assert group_test.shape == (10,)


def test_dataset_y_train_property():
    dataset = Dataset()
    x_train = np.random.rand(2, 5, 1, 10)
    y_train = np.random.rand(5, 1)
    dataset._x_train = x_train
    dataset._y_train = y_train
    y_train_prop = dataset.y_train
    assert y_train_prop.shape == (10, 1)


def test_dataset_invalid_y_train_shape():
    dataset = Dataset()
    x_train = np.random.rand(1, 100, 1, 10)
    dataset._x_train = x_train
    y_train = np.random.rand(100, 2, 2)  # Invalid shape
    with pytest.raises(ValueError):
        dataset.y_train = y_train


def test_dataset_invalid_y_test_shape():
    dataset = Dataset()
    x_test = np.random.rand(1, 50, 1, 10)
    dataset._x_test = x_test
    y_test = np.random.rand(49, 1)
    with pytest.raises(ValueError):
        dataset.y_test = y_test
