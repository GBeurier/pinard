import logging
import re
import numpy as np
import pandas as pd


class WrongFormatError(Exception):
    """Exception raised when X et Y datasets are invalid."""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        msg = ""
        if type(x) is np.ndarray:
            msg += "Invalid X shape: {}".format(x.shape) + " "
        if type(y) is np.ndarray:
            msg += "Invalid Y shape: {}".format(y.shape)
        super().__init__(msg)


def load_csv(
    x_fname,
    y_fname=None,
    y_cols=0,
    *,
    sep=None,
    x_hdr=None,
    y_hdr=None,
    x_index_col=None,
    y_index_col=None,
    autoremove_na=False
):
    """Helper to load a NIRS dataset from csv file(s) using pandas and numpy:
    The data can be either in one file (y_path set to None) or in two files
    for x and y data.
    Rows with NaN or str values are automatically removed.

    Parameters
    ----------
    x_fname : str
        The csv filename for x or xy data.
    y_fname : str, optional
        The csv filename for y data.
    y_cols : int or list of int, optional
        Column index or the list of column index for y data. The column
        index is considered with 'x_start_col' as reference.
        Default: 0
    sep : str, optional
        The csv separator. Default: ';'
    x_hdr : int or list of int or None, optional
        Row number(s) to use as the column names for x data, and the start of the data.
        Default: None.
    y_hdr : int or list of int or None, optional
        Row number(s) to use as the column names for y data, and the start of the data.
        Default: None.
    x_index_col : int, str, sequence of int / str, or False, optional
        Column(s) to use as the row labels of the x data, either given as string name
        or column index.
        Automatically removed from the dataset.
        Default: None.
    y_index_col : int, str, sequence of int / str, or False, optional
        Column(s) to use as the row labels of the y data, either given as string name
        or column index.
        Automatically removed from the dataset.
        Default: None.
    autoremove_na: bool
        The behavior with NA value. 
        If False, raises error if NA values are detected.
        If True, remove rows containing NA values.
        Default: False

    Returns
    -------
    np.array, ,np.array
        Returns x and y data as np.array containing np.float32.
    """
    assert y_fname is not None or y_cols is not None
    # TODO - add assert/exceptions on non-numerical columns
    # TODO - better management of NaN and Null (esp exception msg)

    x_df = pd.read_csv(x_fname, sep=sep, header=x_hdr, index_col=x_index_col,
                       engine="python")
    x_df = x_df.apply(pd.to_numeric, args=("coerce",))

    x_data = x_df.astype(np.float32).values
    x_rows_del = []
    if autoremove_na:
        if np.isnan(x_data).any():
            x_rows_del, _ = np.where(np.isnan(x_data))
            x_data = np.delete(x_data, x_rows_del, axis=0)

    if len(x_data.shape) != 2 or len(x_data) == 0:
        raise WrongFormatError(x_data, None)

    y_data = None
    if y_fname is None:
        y_data = x_data[:, y_cols]
        x_data = np.delete(x_data, y_cols, axis=1)
    else:
        y_df = pd.read_csv(y_fname, sep=sep, header=y_hdr, index_col=y_index_col, engine="python")
        y_df = y_df.apply(pd.to_numeric, args=("coerce",))

        y_data = y_df.astype(np.float32).values
        if autoremove_na:
            if len(x_rows_del) > 0:
                y_data = np.delete(y_data, x_rows_del, axis=0)

            if np.isnan(y_data).any():
                y_rows_del, _ = np.where(np.isnan(y_data))
                y_data = np.delete(y_data, y_rows_del, axis=0)
                x_data = np.delete(x_data, y_rows_del, axis=0)

        if len(y_data.shape) != 2:
            raise WrongFormatError(x_data, y_data)

        if y_cols != -1:
            y_data = y_data[:, y_cols]

    if len(x_data) != len(y_data):
        raise WrongFormatError(x_data, y_data)

    return x_data, y_data


# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# def analyze_csv_file(filename, delimiter_candidates=None, numeric_delimiters=None):
#     """
#     Analyze a CSV file to determine its delimiter, numeric delimiter, and header presence.
#     Parameters can be optionally set by the user.
#     Raises exceptions for various error conditions.
#     """
#     delimiter_candidates = delimiter_candidates or [';', '\t', ',']
#     numeric_delimiters = numeric_delimiters or ['.', ',']

#     try:
#         with open(filename, 'r', newline='') as file:
#             first_lines = [next(file) for _ in range(5)]
#     except Exception as e:
#         logging.error(f"Error reading file: {filename} - {e}")
#         raise

#     delimiter_counts = {delimiter: sum(re.findall(re.escape(delimiter), line) for line in first_lines)
#                         for delimiter in delimiter_candidates}
#     if not any(delimiter_counts.values()):
#         raise ValueError("No delimiter found in the file.")

#     numeric_delimiter_counts = {numeric_delimiter: sum(re.findall(re.escape(numeric_delimiter), line) for line in first_lines[1:])
#                                 for numeric_delimiter in numeric_delimiters}

#     res_delimiter = max(delimiter_counts, key=delimiter_counts.get)
#     res_numeric_delimiter = max(numeric_delimiter_counts, key=numeric_delimiter_counts.get)

#     column_name_pattern = r'^[\w\d_]+$'
#     column_header_present = all(re.match(column_name_pattern, cell.strip()) for cell in first_lines[0].split(res_delimiter))
#     header = 0 if column_header_present else None

#     return res_delimiter, res_numeric_delimiter, header


# def load_csv(file_path, na_policy='auto', delimiter=None, header=None):
#     """
#     Load a CSV file with specified NA handling policy, delimiter, and header.
#     """
#     if not file_path.endswith('.csv'):
#         raise ValueError("File format not supported. Please provide a CSV file.")

#     try:
#         delimiter, nb_delimiter, inferred_header = analyze_csv_file(file_path, delimiter_candidates=[delimiter] if delimiter else None)
#         header = header if header is not None else inferred_header
#         data = pd.read_csv(file_path, header=header, na_filter=False, sep=delimiter, engine='python', skip_blank_lines=False, decimal=nb_delimiter)
#     except Exception as e:
#         logging.error(f"Error loading file: {file_path} - {e}")
#         raise

#     data = data.replace(r"^\s*$", "", regex=True).applymap(lambda x: pd.to_numeric(x, errors='coerce'))

#     data = data.dropna(how='all', axis=1).dropna(how='all', axis=0)
#     rows_with_na = data.isna().any(axis=1)
#     indexes_with_na = data[rows_with_na].index.tolist()

#     if na_policy == 'abort':
#         raise ValueError("NA values found, aborting as per policy.")
#     elif na_policy == 'remove':
#         data = data.drop(indexes_with_na)
#     elif na_policy == 'replace':
#         data = data.fillna(0)
#     elif na_policy == 'auto':
#         threshold = 0.05
#         rows_to_drop = data.index[data.isna().mean(axis=1) > threshold].tolist()
#         data = data.drop(rows_to_drop).fillna(0)
#         indexes_with_na = rows_to_drop

#     data = data.astype(np.float32).values
#     return data, indexes_with_na


# # Example usage
# try:
#     file_path = 'path_to_your_csv_file.csv'
#     data, indexes_with_na = load_csv(file_path, na_policy='auto', delimiter=',', header=0)
# except ValueError as ve:
#     logging.error(ve)
# except Exception as e:
#     logging.error("An unexpected error occurred: %s", e)


# class DatasetFormatError(Exception):
#     """Exception raised for errors in the dataset format."""

#     def __init__(self, message="Dataset format is incorrect"):
#         self.message = message
#         super().__init__(self.message)


# def load_XY_csv(
#     x_fname,
#     y_fname=None,
#     y_cols=0,
#     *,
#     sep=None,
#     x_hdr=None,
#     y_hdr=None,
#     x_index_col=None,
#     y_index_col=None,
#     na_policy='auto'
# ):
#     """
#     Load X and Y datasets from CSV files for deep learning.
#     """
#     if y_fname is None and y_cols is None:
#         raise ValueError("y_cols must be provided if y_fname is None.")

#     # Load X data
#     x_data, x_na_indices = load_csv(x_fname, na_policy=na_policy, delimiter=sep, header=x_hdr)

#     if y_fname is None:
#         if isinstance(y_cols, int):
#             y_cols = [y_cols]
#         if not all(isinstance(col, int) for col in y_cols):
#             raise ValueError("y_cols must be an integer or a list of integers.")

#         if max(y_cols) >= x_data.shape[1]:
#             raise ValueError("y_cols index(es) out of range for the X dataset.")

#         # Extract Y data from the X dataset
#         y_data = x_data[:, y_cols]
#         x_data = np.delete(x_data, y_cols, axis=1)
#     else:
#         # Load Y data
#         y_data, y_na_indices = load_csv(y_fname, na_policy=na_policy, delimiter=sep, header=y_hdr)

#         # Ensure the same rows are removed from both X and Y datasets
#         common_na_indices = set(x_na_indices).intersection(y_na_indices)
#         x_data = np.delete(x_data, list(common_na_indices), axis=0)
#         y_data = np.delete(y_data, list(common_na_indices), axis=0)

#         if isinstance(y_cols, int):
#             y_data = y_data[:, [y_cols]]

#     # Check if X and Y datasets are compliant for training
#     if len(x_data) != len(y_data):
#         raise DatasetFormatError("X and Y datasets have different row counts.")

#     return x_data, y_data


# # Example usage
# try:
#     x_data, y_data = load_XY_csv('path_to_x.csv', y_fname=None, y_cols=0, sep=';', x_hdr=0)
# except DatasetFormatError as dfe:
#     logging.error(dfe)
# except ValueError as ve:
#     logging.error(ve)
# except Exception as e:
#     logging.error("An unexpected error occurred: %s", e)
