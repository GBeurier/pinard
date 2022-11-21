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
    sep=";",
    x_hdr=None,
    y_hdr=None,
    x_index_col=None,
    y_index_col=None,
    remove_na=False
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
    remove_na: bool
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

    x_df = pd.read_csv(x_fname, sep=sep, header=x_hdr, index_col=x_index_col)
    x_df = x_df.apply(pd.to_numeric, args=("coerce",))
    if remove_na:
        x_df = x_df.dropna()
    elif x_df.isna().values.any():
        raise WrongFormatError(x_df, None)

    x_data = x_df.astype(np.float32).values
    if len(x_data.shape) != 2 or len(x_data) == 0:
        raise WrongFormatError(x_data, None)

    y_data = None
    if y_fname is None:
        y_data = x_data[:, y_cols]
        x_data = np.delete(x_data, y_cols, axis=1)
    else:
        y_df = pd.read_csv(y_fname, sep=sep, header=y_hdr, index_col=y_index_col)
        y_df = y_df.apply(pd.to_numeric, args=("coerce",))
        if remove_na:
            y_df = y_df.dropna()
        elif y_df.isna().values.any():
            raise WrongFormatError(None, x_df)
            
        y_data = y_df.astype(np.float32).values

        if len(y_data.shape) != 2:
            raise WrongFormatError(x_data, y_data)

        if y_cols != -1:
            y_data = y_data[:, y_cols]

    if len(x_data) != len(y_data):
        raise WrongFormatError(x_data, y_data)

    return x_data, y_data
