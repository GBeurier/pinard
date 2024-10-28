# tests/test_dataset_loader.py

import pytest
from pinard.data.dataset_loader import (
    _merge_params,
    load_XY,
    handle_data,
    get_dataset,
)
import numpy as np
from pathlib import Path


def test_merge_params():
    global_params = {'a': 1, 'b': 2}
    handler_params = {'b': 3, 'c': 4}
    local_params = {'c': 5, 'd': 6}
    result = _merge_params(local_params, handler_params, global_params)
    expected = {'a': 1, 'b': 3, 'c': 5, 'd': 6}
    assert result == expected


# def test_load_XY(tmp_path):
#     x_content = """1,2
# 3,4
# 5,6
# 7,8
# 9,10"""
#     y_content = """10
# 20
# 30
# 40
# 50"""
#     x_file = tmp_path / "x.csv"
#     y_file = tmp_path / "y.csv"
#     x_file.write_text(x_content)
#     y_file.write_text(y_content)
#     x_params = {}
#     y_params = {}
#     x, y = load_XY(str(x_file), None, x_params, str(y_file), None, y_params)
#     expected_x = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=np.float32)
#     expected_y = np.array([[10], [20], [30], [40], [50]], dtype=np.float32)
#     assert np.array_equal(x, expected_x)
#     assert np.array_equal(y, expected_y)


def test_load_XY_invalid_x():
    x_params = {}
    y_params = {}
    with pytest.raises(ValueError):
        load_XY(None, None, x_params, None, None, y_params)


def test_load_XY_invalid_y(tmp_path):
    x_content = """1,2
3,4
5,6
7,8
9,10"""
    x_file = tmp_path / "x.csv"
    x_file.write_text(x_content)
    x_params = {}
    y_params = {}
    with pytest.raises(ValueError):
        load_XY(str(x_file), None, x_params, None, None, y_params)


# def test_load_XY_y_from_x(tmp_path):
#     x_content = """1,2,10
# 3,4,20
# 5,6,30
# 7,8,40
# 9,10,50"""
#     x_file = tmp_path / "x.csv"
#     x_file.write_text(x_content)
#     x_params = {}
#     y_params = {}
#     x_filter = None
#     y_filter = [2]
#     x, y = load_XY(str(x_file), x_filter, x_params, None, y_filter, y_params)
#     expected_x = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=np.float32)
#     expected_y = np.array([[10], [20], [30], [40], [50]], dtype=np.float32)
#     assert np.array_equal(x, expected_x)
#     assert np.array_equal(y, expected_y)


# def test_handle_data(tmp_path):
#     x_content = """1,2
# 3,4
# 5,6
# 7,8
# 9,10"""
#     y_content = """10
# 20
# 30
# 40
# 50"""
#     x_file = tmp_path / "x.csv"
#     y_file = tmp_path / "y.csv"
#     x_file.write_text(x_content)
#     y_file.write_text(y_content)
#     config = {
#         'train_x': str(x_file),
#         'train_y': str(y_file),
#     }
#     x, y = handle_data(config, 'train')
#     expected_x = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=np.float32)
#     expected_y = np.array([[10], [20], [30], [40], [50]], dtype=np.float32)
#     assert np.array_equal(x, expected_x)
#     assert np.array_equal(y, expected_y)


# def test_get_dataset(tmp_path):
#     x_content = """1,2
# 3,4
# 5,6
# 7,8
# 9,10"""
#     y_content = """10
# 20
# 30
# 40
# 50"""
#     x_file = tmp_path / "x.csv"
#     y_file = tmp_path / "y.csv"
#     x_file.write_text(x_content)
#     y_file.write_text(y_content)
#     data_config = {
#         'train_x': str(x_file),
#         'train_y': str(y_file),
#         'test_x': str(x_file),
#         'test_y': str(y_file),
#     }
#     dataset = get_dataset(data_config)
#     assert dataset.x_train.shape == (1, 5, 1, 2)
#     assert dataset.y_train_init.shape == (5, 1)
#     assert dataset.x_test.shape == (1, 5, 1, 2)
#     assert dataset.y_test_init.shape == (5, 1)


# def test_get_dataset_invalid_config():
#     data_config = {}
#     with pytest.raises(ValueError):
#         dataset = get_dataset(data_config)
