# dataset_loader.py

from pathlib import Path
import hashlib
import json
from .data_config_parser import parse_config
from .dataset import Dataset
from .csv_loader import load_csv
import numpy as np


def _merge_params(local_params, handler_params, global_params):
    """
    Merge parameters from local, handler, and global scopes.

    Parameters:
    - local_params (dict): Local parameters specific to the data subset.
    - handler_params (dict): Parameters specific to the handler.
    - global_params (dict): Global parameters that apply to all handlers.

    Returns:
    - dict: Merged parameters with precedence: local > handler > global.
    """
    merged_params = {} if global_params is None else global_params.copy()
    if handler_params is not None:
        merged_params.update(handler_params)
    if local_params is not None:
        merged_params.update(local_params)
    return merged_params


def load_XY(x_path, x_filter, x_params, y_path, y_filter, y_params):
    """
    Load X and Y data from the given paths, apply filters, and return numpy arrays.

    Parameters:
    - x_path (str): Path to the X data file.
    - x_filter: Filter to apply to X data (not implemented yet).
    - x_params (dict): Parameters for loading X data.
    - y_path (str): Path to the Y data file (can be None).
    - y_filter: Filter to apply to Y data (or indices if y_path is None).
    - y_params (dict): Parameters for loading Y data.

    Returns:
    - tuple: (x, y) where x and y are numpy arrays.

    Raises:
    - ValueError: If data is invalid or if there are inconsistencies.
    """
    if x_path is None:
        raise ValueError("Invalid x definition: x_path is None")

    x, report = load_csv(x_path, **x_params)
    print(report)

    if "error" in report:
        raise ValueError(f"Invalid data: x contains errors: {report['error']}")

    if x is None:
        raise ValueError("Invalid data: x is None")

    if x_filter is not None:
        raise NotImplementedError("Auto-filtering not implemented yet")

    if y_path is None:
        # Y is a subset of X
        if y_filter is None:
            raise ValueError("Invalid y definition: y_path and y_filter are both None")

        if not all(isinstance(i, int) for i in y_filter):
            raise ValueError("Invalid y definition: y_filter is not a list of integers. Other filters not implemented yet")

        if any(i < 0 or i >= x.shape[1] for i in y_filter):
            raise ValueError("Invalid y definition: y_filter contains invalid indices")

        y = x[:, y_filter]
        x = x[:, [i for i in range(x.shape[1]) if i not in y_filter]]

    else:
        # Y is in a separate file
        y, report = load_csv(y_path, na_policy=y_params.get('na_policy', 'auto'), type="y", **y_params)
        
        if "error" in report:
            raise ValueError(f"Invalid data: y contains errors: {report['error']}")

        if y is None:
            raise ValueError("Invalid data: y is None")

        if y_filter is not None:
            raise NotImplementedError("Auto-filtering not implemented yet")

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Invalid data: x and y have different number of rows ({x.shape[0]} != {y.shape[0]})")

    return x, y


def id_config(config, t_set, subset, params):
    """
    Generate a unique ID for the data configuration based on the provided parameters.

    Parameters:
    - config (dict): Data configuration dictionary.
    - t_set (str): The dataset type ('train', 'valid', 'test').
    - subset (str): 'x' or 'y'.
    - params (dict): Parameters used for loading the data.

    Returns:
    - str: Unique identifier for the data configuration.
    """
    keys_to_extract = [f'{t_set}_{subset}', f'{t_set}_{subset}_filter']
    if config.get(f'{t_set}_{subset}') is None:
        if subset == 'x' or (subset == 'y' and config.get(f'{t_set}_x') is None):
            keys_to_extract.extend(['train_x', 'train_x_filter', 'train_x_params'])
        elif subset == 'y':
            keys_to_extract.extend([f'{t_set}_x', f'{t_set}_x_filter', f'{t_set}_x_params'])

    # Build a subset of the config for hashing
    subset_config = {key: config.get(key) for key in set(keys_to_extract)}
    subset_config['params'] = params

    # Create a string representation and generate MD5 hash
    config_str = json.dumps(subset_config, sort_keys=True)
    id_hash = hashlib.md5(config_str.encode()).hexdigest()[0:8]

    return id_hash


def handle_data(config, t_set):
    """
    Handle data loading and caching for a given dataset type (train, test).

    Parameters:
    - config (dict): Data configuration dictionary.
    - t_set (str): The dataset type ('train', 'test').

    Returns:
    - tuple: (x_id, y_id) cache IDs for X and Y data.
    """
    x_params = _merge_params(config.get(f'{t_set}_x_params'), config.get(f'{t_set}_params'), config.get('global_params'))
    y_params = _merge_params(config.get(f'{t_set}_y_params'), config.get(f'{t_set}_params'), config.get('global_params'))
    x, y = load_XY(config.get(f'{t_set}_x'), config.get(f'{t_set}_x_filter'), x_params,
                   config.get(f'{t_set}_y'), config.get(f'{t_set}_y_filter'), y_params)
    return x, y


def get_dataset(data_config):
    """
    Load dataset based on the data configuration.

    Parameters:
    - data_config: Data configuration (can be a dict or a path to a config file).

    Returns:
    - Dataset: Dataset object with loaded data IDs.
    """
    config = parse_config(data_config)
    x_train, y_train = handle_data(config, "train")
    x_test, y_test = handle_data(config, "test")
    dataset = Dataset()
    dataset.x_train = x_train
    dataset.y_train_init = y_train
    dataset.x_test = x_test
    dataset.y_test_init = y_test
    

    # Handle training data
    # x_train_id, y_train_id = handle_data(config, "train")
    # if x_train_id is None:
        # raise ValueError("Invalid data: train_x is None")

    # Handle validation/test data
    # x_test_id, y_test_id = handle_data(config, "valid")  # TODO: Uniformize test/valid on files

    # datatuple = DataTuple(
    #     x_train_id=x_train_id,
    #     y_train_id=y_train_id,
    #     x_test_id=x_test_id,
    #     y_test_id=y_test_id,
    # )
    # dataset.set_data_tuple("raw", datatuple)
    
    return dataset

