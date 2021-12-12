import pandas as pd
import numpy as np

from dataclasses import dataclass
from typing import Type


### spectra bounds     ## 350 - 2500 (NIR) / 25000 (IR)


@dataclass
class Spectra:
    x: np.ndarray
    y: np.ndarray
    proc_x: np.ndarray
    
    def __init__(self):
        pass
    #     self.x = np.array()
    #     self.y = np.array()
    #     self.proc_x = np.array()


class NIRS_Set:
    """ A class to manage a set of spectrum.
        It provides tools to load spectra (X) and analysis (Y), to apply filter methods and to manage data.
    """
    
    def __init__(self, name = ""):
        self.name = name
        self.set = Spectra()
        # self.augmented_set = Spectra()
        # self.scaled_set = Spectra()
    
        
       
    def load(self, x_path, y_path = None, y_cols = None, x_hdr = None, y_hdr = None):
        assert(y_path != None or y_cols != None)
        
        x_data = pd.read_csv(x_path, sep = ";", header = x_hdr, dtype=np.float32).values
        assert not(np.isnan(x_data).any())
        
        y_data = None
        if y_path == None:
            y_data = x_data[:, y_cols]
            x_data = np.delete(x_data, y_cols, axis = 1)
        else:
            y_data = pd.read_csv(y_path, sep = ";", header = y_hdr).values
            assert not(np.isnan(y_data).any())
            if y_cols != -1:
                y_data = y_data[:, y_cols]
        
        if y_cols != 1:    
            if isinstance(y_cols, list):
                y_data = y_data.reshape((len(y_data), len(y_cols)))
            else:
                y_data = y_data.reshape((len(y_data), 1))
        
        
        ## TODO ensure that x_data width is even number otherwise duplicates last col
        self.set.x = x_data
        self.set.y = y_data
    
    def get_raw_x(self):
        return self.set.x
    
    def get_raw_y(self):
        return self.set.y
    

    def augment(self, ):
        pass
    
    def preprocess(self):
        pass
    
    def partial_fit(self):
        pass
    
    def transform(self):
        pass

# def resample(wavelength, spectra, resampling_ratio):
#     """ Resample spectra according to the resampling ratio.
#     Args:
#         wavelength <numpy.ndarray>: Vector of wavelengths.
#         spectra <numpy.ndarray>: NIRS data matrix.
#         resampling_ratio <float>: new length with respect to original length
#     Returns:
#         wavelength_ <numpy.ndarray>: Resampled wavelengths.
#         spectra_ <numpy.ndarray>: Resampled NIR spectra
#     """

#     new_length = int(np.round(wavelength.size * resampling_ratio))
#     spectra_, wavelength_ = scipy.signal.resample(spectra, new_length, wavelength)
#     return wavelength_, spectra_


# def clip(wavelength, spectra, threshold, substitute=None):
#     """ Removes or substitutes values above the given threshold.
#     Args:
#         wavelength <numpy.ndarray>: Vector of wavelengths.
#         spectra <numpy.ndarray>: NIRS data matrix.
#         threshold <float>: threshold value for rejection
#         substitute <float>: substitute value for rejected values (None removes values from the spectra)
#     Returns:
#         wavelength <numpy.ndarray>: Vector of wavelengths.
#         spectra <numpy.ndarray>: NIR spectra with threshold exceeding values removed.
#     """

#     if substitute == None:  # remove threshold violations
#         mask = np.any(spectra > threshold, axis=1)
#         spectra = spectra[~mask, :]
#         wavelength = wavelength[~mask]
#     else:  # substitute threshold violations with a value
#         spectra[spectra > threshold] = substitute
#     return wavelength, spectra

#     return wavelength, spectra
