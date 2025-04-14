# dataset_loader.py

from pathlib import Path
import hashlib
import json
from .data_config_parser import parse_config
from .dataset import Dataset
from .csv_loader import load_csv

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
    - tuple: (x, y, x_report, y_report) where x and y are numpy arrays and reports contain metadata.

    Raises:
    - ValueError: If data is invalid or if there are inconsistencies.
    """
    if x_path is None:
        raise ValueError("Invalid x definition: x_path is None")

    # Default to 'auto' for categorical detection if not specified
    if 'categorical_mode' not in x_params:
        x_params['categorical_mode'] = 'auto'
    if 'data_type' not in x_params:
        x_params['data_type'] = 'x'

    x, x_report = load_csv(x_path, **x_params)

    if "error" in x_report and x_report["error"] is not None:
        raise ValueError(f"Invalid data: x contains errors: {x_report['error']}")

    if x is None:
        raise ValueError("Invalid data: x is None")

    if x_filter is not None:
        raise NotImplementedError("Auto-filtering not implemented yet")
        
    y_report = {"error": None, "categorical_info": {}, "warnings": []}

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
        
        # Handle categorical columns if present in extracted y
        # Not implemented for this case yet as it would require tracking column names

    else:
        # Y is in a separate file
        # Remove the 'na_policy' from y_params as we're passing it explicitly
        y_params_copy = y_params.copy()
        na_policy = y_params_copy.pop('na_policy', 'auto')
        
        # Set categorical mode and data type for y file
        if 'categorical_mode' not in y_params_copy:
            y_params_copy['categorical_mode'] = 'auto'
        if 'data_type' not in y_params_copy:
            y_params_copy['data_type'] = 'y'
            
        y, y_report = load_csv(y_path, na_policy=na_policy, **y_params_copy)

        if "error" in y_report and y_report["error"] is not None:
            raise ValueError(f"Invalid data: y contains errors: {y_report['error']}")

        if y is None:
            raise ValueError("Invalid data: y is None")

        # Print warnings about categorical columns if any were detected
        if y_report.get("warnings"):
            for warning in y_report["warnings"]:
                print(f"Warning: {warning}")

        if y_filter is not None:
            raise NotImplementedError("Auto-filtering not implemented yet")

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Invalid data: x and y have different number of rows ({x.shape[0]} != {y.shape[0]})")

    return x, y, x_report, y_report


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
    - tuple: (x, y, x_report, y_report) data and metadata reports
    """
    if config is None:
        raise ValueError(f"Configuration for {t_set} dataset is None")

    x_params = _merge_params(config.get(f'{t_set}_x_params'), config.get(f'{t_set}_params'), config.get('global_params'))
    y_params = _merge_params(config.get(f'{t_set}_y_params'), config.get(f'{t_set}_params'), config.get('global_params'))
    x, y, x_report, y_report = load_XY(config.get(f'{t_set}_x'), config.get(f'{t_set}_x_filter'), x_params,
                       config.get(f'{t_set}_y'), config.get(f'{t_set}_y_filter'), y_params)
    return x, y, x_report, y_report


def get_dataset(data_config):
    """
    Load dataset based on the data configuration.
    
    Parameters:
    - data_config: Data configuration (can be a dict or a path to a config file).
    
    Returns:
    - Dataset: Dataset object with loaded data and metadata.
    """
    config = parse_config(data_config)
    if config is None:
        raise ValueError("Dataset configuration is None")

    dataset = Dataset()
    try:
        x_train, y_train, x_train_report, y_train_report = handle_data(config, "train")
        x_test, y_test, x_test_report, y_test_report = handle_data(config, "test")
        
        dataset.x_train = x_train
        dataset.y_train_init = y_train
        dataset.x_test = x_test
        dataset.y_test_init = y_test
        
        # Store categorical information if present
        if y_train_report and y_train_report.get("categorical_info"):
            dataset.y_train_categorical_info = y_train_report["categorical_info"]
            
        if y_test_report and y_test_report.get("categorical_info"):
            dataset.y_test_categorical_info = y_test_report["categorical_info"]
            
    except Exception as e:
        print("Error loading data:", e)
        raise

    return dataset

