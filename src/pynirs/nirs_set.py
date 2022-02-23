import pandas as pd
import numpy as np

from dataclasses import dataclass
from typing import Type

import preprocessor


### spectra bounds     ## 350 - 2500 (NIR) / 25000 (IR)
PREPROCESSING = ['x', 'x*detrend', 'x*snv', 'x*rnv', 'x*savgol1', 'x*msc', 'x*derivate', 'x*gaussian1', 'x*gaussian2', 'x*wv_haar', 'x*detrend*snv', 'x*detrend*rnv', 'x*detrend*savgol1', 'x*detrend*msc', 'x*detrend*derivate', 'x*detrend*gaussian1', 'x*detrend*gaussian2', 'x*detrend*wv_haar', 'x*wv_bior2.2', 'x*wv_bior3.1', 'x*wv_bior4.4', 'x*wv_bior5.5', 'x*wv_bior6.8', 'x*wv_coif1', 'x*wv_coif2', 'x*wv_coif3', 'x*wv_coif4', 'x*wv_coif5', 'x*wv_coif6', 'x*wv_coif7', 'x*wv_coif8', 'x*wv_coif9', 'x*wv_coif10', 'x*wv_coif11', 'x*wv_coif12', 'x*wv_coif13', 'x*wv_coif14', 'x*wv_coif15', 'x*wv_coif16', 'x*wv_coif17', 'x*wv_db2', 'x*wv_db3', 'x*wv_db4', 'x*wv_db5', 'x*wv_db6', 'x*wv_db7', 'x*wv_db8', 'x*wv_db9', 'x*wv_db10', 'x*wv_db11', 'x*wv_db12', 'x*wv_db13', 'x*wv_db14', 'x*wv_db15', 'x*wv_db16', 'x*wv_db17', 'x*wv_db18', 'x*wv_db19', 'x*wv_db20', 'x*wv_db21', 'x*wv_db22', 'x*wv_db23', 'x*wv_db24', 'x*wv_db25', 'x*wv_db26', 'x*wv_db27', 'x*wv_db28', 'x*wv_db29', 'x*wv_db30', 'x*wv_db31', 'x*wv_db32', 'x*wv_db33', 'x*wv_db34', 'x*wv_db35', 'x*wv_db36', 'x*wv_db37', 'x*wv_db38', 'x*wv_dmey', 'x*wv_rbio2.2', 'x*wv_rbio3.1', 'x*wv_rbio4.4', 'x*wv_rbio5.5', 'x*wv_rbio6.8', 'x*wv_sym4', 'x*wv_sym5', 'x*wv_sym6', 'x*wv_sym7', 'x*wv_sym8', 'x*wv_sym9', 'x*wv_sym10', 'x*wv_sym11', 'x*wv_sym12', 'x*wv_sym13', 'x*wv_sym14', 'x*wv_sym15', 'x*wv_sym16', 'x*wv_sym17', 'x*wv_sym18', 'x*wv_sym19', 'x*wv_sym20', 'x*snv*snv', 'x*rnv*snv', 'x*savgol1*snv', 'x*msc*snv', 'x*derivate*snv', 'x*gaussian1*snv', 'x*gaussian2*snv', 'x*wv_haar*snv', 'x*snv*rnv', 'x*rnv*rnv', 'x*savgol1*rnv', 'x*msc*rnv', 'x*derivate*rnv', 'x*gaussian1*rnv', 'x*gaussian2*rnv', 'x*wv_haar*rnv', 'x*snv*savgol1', 'x*rnv*savgol1', 'x*savgol1*savgol1', 'x*msc*savgol1', 'x*derivate*savgol1', 'x*gaussian1*savgol1', 'x*gaussian2*savgol1', 'x*wv_haar*savgol1', 'x*snv*msc', 'x*rnv*msc', 'x*savgol1*msc', 'x*msc*msc', 'x*derivate*msc', 'x*gaussian1*msc', 'x*gaussian2*msc', 'x*wv_haar*msc', 'x*snv*derivate', 'x*rnv*derivate', 'x*savgol1*derivate', 'x*msc*derivate', 'x*derivate*derivate', 'x*gaussian1*derivate', 'x*gaussian2*derivate', 'x*wv_haar*derivate', 'x*snv*gaussian1', 'x*rnv*gaussian1', 'x*savgol1*gaussian1', 'x*msc*gaussian1', 'x*derivate*gaussian1', 'x*gaussian1*gaussian1', 'x*gaussian2*gaussian1', 'x*wv_haar*gaussian1', 'x*snv*gaussian2', 'x*rnv*gaussian2', 'x*savgol1*gaussian2', 'x*msc*gaussian2', 'x*derivate*gaussian2', 'x*gaussian1*gaussian2', 'x*gaussian2*gaussian2', 'x*wv_haar*gaussian2', 'x*snv*wv_haar', 'x*rnv*wv_haar', 'x*savgol1*wv_haar', 'x*msc*wv_haar', 'x*derivate*wv_haar', 'x*gaussian1*wv_haar', 'x*gaussian2*wv_haar', 'x*wv_haar*wv_haar']

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
    
        
       
    def load(self, x_path, y_path = None, y_cols = None, *, x_hdr = None, y_hdr = None):
        assert(y_path != None or y_cols != None)
        
        x_df = pd.read_csv(x_path, sep = ";", header = x_hdr, dtype=np.float32)
        x_data = x_df.values
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
        
        # if y_cols != 1:    
        #     if isinstance(y_cols, list):
        #         y_data = y_data.reshape((len(y_data), len(y_cols)))
        #     else:
        #         y_data = y_data.reshape((len(y_data), 1))
        
        
        ## TODO ensure that x_data width is even number otherwise duplicates last col
        self.set.x = x_data
        self.set.y = y_data
        return x_data, y_data
    
    def preprocess(self):
        self.set.proc_x = preprocessor.process(self.set.x, PREPROCESSING)
    
    def get_raw_x(self):
        return self.set.x
    
    def get_raw_y(self):
        return self.set.y
    

    def augment(self, ):
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
