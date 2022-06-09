import pandas as pd
import numpy as np

from dataclasses import dataclass
from typing import Type


@dataclass
class Spectra:
    x: np.ndarray
    y: np.ndarray
    proc_x: np.ndarray

    def __init__(self):
        pass

class NIRS_Set:
    """ A class to manage a set of spectrum.
        It provides tools to load spectra (X) and analysis (Y), to apply filter methods and to manage data.
    """

    def __init__(self, name=""):
        self.name = name
        self.set = Spectra()

    def load(self, x_path, y_path=None, y_cols=None, *, x_hdr=None, y_hdr=None):
        assert(y_path != None or y_cols != None)

        x_df = pd.read_csv(x_path, sep=";", header=x_hdr, dtype=np.float32)
        x_data = x_df.values
        assert not(np.isnan(x_data).any())

        y_data = None
        if y_path == None:
            y_data = x_data[:, y_cols]
            x_data = np.delete(x_data, y_cols, axis=1)
        else:
            y_data = pd.read_csv(y_path, sep=";", header=y_hdr).values
            assert not(np.isnan(y_data).any())
            if y_cols != -1:
                y_data = y_data[:, y_cols]

        # TODO ensure that x_data width is even number otherwise duplicates last col
        self.set.x = x_data
        self.set.y = y_data
        return x_data, y_data

    def get_raw_x(self):
        return self.set.x

    def get_raw_y(self):
        return self.set.y
