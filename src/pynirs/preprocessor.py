from scipy import (
    sparse,
    signal,
    ndimage
)

import warnings
import numpy as np

from sklearn.preprocessing import StandardScaler as StandardNormalVariate
from sklearn.preprocessing import RobustScaler as RobustNormalVariate
from sklearn.preprocessing import scale
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import (
    check_is_fitted, 
    # check_random_state, 
    # _check_sample_weight, 
    FLOAT_DTYPES, 
)
# import pywt
# import threading, queue

# #np.mean, np.std, np.median, np.percentile([lower, upper]), np.max(), np.min(), np.linalg.norm(), np.polyfit()

__all__ = [
    "Baseline", 
    "StandardNormalVariate", ## SAME as STANDARDSCALER
    "RobustNormalVariate", ## SAME as ROBUSTSCALER
    "SavitzkyGolay",
    "Normalize",
    "Detrend",
    "MultiplicativeScatterCorrection",
    "ExtendedMultiplicativeScatterCorrection",
    "baseline", 
    # "snv", 
    # "rnv",
    "savgol",
    "norml",
    "detrend",
    "msc",
    "emsc",
]


class Baseline(TransformerMixin, BaseEstimator):
    
    def __init__(self, *, copy = True):
        self.copy = copy

    def _reset(self):
        if hasattr(self, "mean_"):
            del self.mean_

    def fit(self, X, y = None):
        self._reset()        
        return self.partial_fit(X, y)
    
    def partial_fit(self, X, y = None):
        if sparse.issparse(X):
            raise TypeError("Baseline does not support sparse input")
        
        first_pass = not hasattr(self, "mean_")
        X = self._validate_data(
            X, 
            reset = first_pass, 
            dtype = FLOAT_DTYPES, 
            estimator = self
        )
        
        self.mean_ = np.mean(X, axis = 0)
        return self

    def transform(self, X):
        check_is_fitted(self)
        
        X = self._validate_data(
            X, 
            reset = False, 
            copy = self.copy, 
            dtype = FLOAT_DTYPES, 
            estimator = self
        )
        
        X = X - self.mean_
        return X
    
    def inverse_transform(self, X):
        check_is_fitted(self)

        X = check_array(
            X, 
            copy = self.copy, 
            dtype = FLOAT_DTYPES
        )

        X = X + self.mean_
        return X
    
    def _more_tags(self):
        return {'allow_nan': False}


def baseline(spectra):
    """ Removes baseline (mean) from each spectrum.
    Args:
        spectra < numpy.ndarray > : NIRS data matrix.
    Returns:
        spectra < numpy.ndarray > : Mean-centered NIRS data matrix
    """

    return spectra - np.mean(spectra, axis = 0)




class SavitzkyGolay(TransformerMixin, BaseEstimator):
    
    def __init__(self, filter_win = 11, poly_order = 3, deriv_order = 0, delta = 1.0, *, copy = True):
        self.copy = copy
        self.filter_win = filter_win
        self.poly_order = poly_order
        self.deriv_order = deriv_order
        self.delta = delta

    def _reset(self):
        pass

    def fit(self, X, y = None):
        if sparse.issparse(X):
            raise ValueError("SavitzkyGolay does not support sparse input")
        return self

    def transform(self, X, copy = None):
        if sparse.issparse(X):
            raise ValueError('Sparse matrices not supported!"')

        X = self._validate_data(
            X, 
            reset = False, 
            copy = self.copy, 
            dtype = FLOAT_DTYPES, 
            estimator = self
        )
                
        X = signal.savgol_filter(X.T, filter_win = self.filter_win, poly_order = self.poly_order, deriv_order = self.deriv_order, delta = self.delta)
        return X

    def _more_tags(self):
        return {'allow_nan': False}
    
    

def savgol(spectra, filter_win = 11, poly_order = 3, deriv_order = 0, delta = 1.0):
    """ Perform Savitzky–Golay filtering on the data (also calculates derivatives). This function is a wrapper for
    scipy.signal.savgol_filter.
    Args:
        spectra < numpy.ndarray > : NIRS data matrix.
        filter_win < int > : Size of the filter window in samples (default 11).
        poly_order < int > : Order of the polynomial estimation (default 3).
        deriv_order < int > : Order of the derivation (default 0).
    Returns:
        spectra < numpy.ndarray > : NIRS data smoothed with Savitzky-Golay filtering
    """
    return signal.savgol_filter(spectra, filter_win, poly_order, deriv_order, delta = delta)



class Normalize(TransformerMixin, BaseEstimator):
    """Normalize spectrum using either custom range of linalg normalization
    
    Parameters
    ----------
    feature_range : tuple (min, max), default=(-1, -1)
        Desired range of transformed data. If range min and max equals -1, linalg 
        normalization is applied, otherwise user defined normalization 
        is applied
    copy : bool, default=True
        Set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array).

    """

    def __init__(self, feature_range = (-1, -1), *, copy = True):
        self.copy = copy
        self.feature_range = feature_range
        self.user_defined =  (feature_range[0] != -1 and feature_range[1] != -1)

    def _reset(self):
        if hasattr(self, "min_"):
            del self.min_
            del self.max_
            del self.f_
            
        if hasattr(self, "linalg_norm_"):
            del self.linalg_norm_

    def fit(self, X, y = None):
        self._reset()        
        return self.partial_fit(X, y)
    
    def partial_fit(self, X, y = None):
        feature_range = self.feature_range
        if self.user_defined and feature_range[0] >= feature_range[1]:
            warnings.warn(
                "Minimum of desired feature range should be smaller than maximum. Got %s."
                % str(feature_range),
                SyntaxWarning
            )
        
        if self.user_defined and feature_range[0] == feature_range[1]:
            raise ValueError(
                "Feature range is not correctly defined. Got %s."
                % str(feature_range)
            )
            
        if sparse.issparse(X):
            raise TypeError("Normalization does not support sparse input")
        
        first_pass = not hasattr(self, "min_")
        X = self._validate_data(
            X, 
            reset = first_pass, 
            dtype = FLOAT_DTYPES, 
            estimator = self
        )
        
        if self.user_defined:
            self.min_ = np.min(X, axis = 0)
            self.max_ = np.max(X, axis = 0)
            imin = self.feature_range[0]
            imax = self.feature_range[1]
            self.f_ = (imax - imin)/(self.max_ - self.min_)
        else:
            self.linalg_norm_ = np.linalg.norm(X, axis = 0)
        return self

    def transform(self, X):
        check_is_fitted(self)
        
        X = self._validate_data(
            X, 
            reset = False, 
            copy = self.copy, 
            dtype = FLOAT_DTYPES, 
            estimator = self
        )
        
        if self.user_defined:
            imin = self.feature_range[0]
            f = self.f_
            n = X.shape
            arr = np.empty((0, n[0]), dtype = float)
            for i in range(0, n[1]):
                d = X[:, i]
                dnorm = imin + f*d
                arr = np.append(arr, [dnorm], axis = 0)
            X = np.transpose(arr)
        else:
            X = X / self.linalg_norm_
        return X
    
    def inverse_transform(self, X):
        check_is_fitted(self)

        X = check_array(
            X, 
            copy = self.copy, 
            dtype = FLOAT_DTYPES
        )

        if self.user_defined:
            imin = self.feature_range[0]
            f = self.f_
            n = X.shape
            arr = np.empty((0, n[0]), dtype = float)
            for i in range(0, n[1]):
                d = X[:, i]
                dnorm = d/f - imin
                arr = np.append(arr, [dnorm], axis = 0)
            X = np.transpose(arr)
        else:
            X = X * self.linalg_norm_
        return X
        
    def _more_tags(self):
        return {'allow_nan': False}


def norml(spectra, feature_range = (-1, -1)):
    """ Perform spectral normalisation with user-defined limits. (numpy linalg)
    Args:
        spectra < numpy.ndarray > : NIRS data matrix.
        feature_range : tuple (min, max), default=(-1, -1)
            Desired range of transformed data. If range min and max equals -1, linalg 
            normalization is applied, otherwise user bounds defined normalization 
            is applied
    Returns:
        spectra < numpy.ndarray > : Normalized NIR spectra
    """
    if feature_range[0] != -1 and feature_range[0] != -1:
        imin = feature_range[0]
        imax = feature_range[1]
        if imin >= imax:
            warnings.warn(
                "Minimum of desired feature range should be smaller than maximum. Got %s."
                % str(feature_range),
                SyntaxWarning
            )
        if imin == imax:
            raise ValueError(
                "Feature range is not correctly defined. Got %s."
                % str(feature_range)
            )
            
        f = (imax - imin)/(np.max(spectra) - np.min(spectra))
        n = spectra.shape
        arr = np.empty((0, n[0]), dtype = float) #create empty array for spectra
        for i in range(0, n[1]):
            d = spectra[:, i]
            dnorm = imin + f*d
            arr = np.append(arr, [dnorm], axis = 0)
        return np.transpose(arr)
    else:
        return spectra / np.linalg.norm(spectra, axis = 0)


class Detrend(TransformerMixin, BaseEstimator):
    
    def __init__(self, bp=0, *, copy=True):
        self.copy = copy
        self.bp = bp

    def _reset(self):
        pass
    
    def fit(self, X, y=None):
        if sparse.issparse(X):
            raise ValueError('Sparse matrices not supported!"')
        return self

    def transform(self, X, copy=None):
        if sparse.issparse(X):
            raise ValueError('Sparse matrices not supported!"')
        X = self._validate_data(
            X, 
            reset = False, 
            copy = self.copy, 
            dtype = FLOAT_DTYPES, 
            estimator = self
        )
        
        X = signal.detrend(X, bp = self.bp)
        return X

    def _more_tags(self):
        return {'allow_nan': False}



def detrend(spectra, bp = 0):
    """ Perform spectral detrending to remove linear trend from data.
    Args:
        spectra < numpy.ndarray > : NIRS data matrix.
        bp < list > : A sequence of break points. If given, an individual linear fit is performed for each part of data
        between two break points. Break points are specified as indices into data.
    Returns:
        spectra < numpy.ndarray > : Detrended NIR spectra
    """
    return signal.detrend(spectra, bp = bp)


class MultiplicativeScatterCorrection(TransformerMixin, BaseEstimator):

    def __init__(self, scale = True, *, copy = True):
        self.copy = copy
        self.scale = scale

    def _reset(self):
        if hasattr(self, "scaler_"):
            del self.scaler_
            del self.a_
            del self.b_
     
    def fit(self, X, y = None):
        self._reset()        
        return self.partial_fit(X, y)
    
    def partial_fit(self, X, y = None):
        
        if sparse.issparse(X):
            raise TypeError("Normalization does not support sparse input")
        
        first_pass = not hasattr(self, "mean_")
        X = self._validate_data(
            X, 
            reset = first_pass, 
            dtype = FLOAT_DTYPES, 
            estimator = self
        )
        
        tmp_x = X
        if self.scale:
            scaler = StandardScaler()
            scaler.fit(X)
            self.scaler_ = scaler
            tmp_x = scaler.transform(X, with_std = False, axis = 0)
        
        
        reference = np.mean(tmp_x, axis = 1)
        
        a = np.empty(X.shape[1], dtype = float)
        b = np.empty(X.shape[1], dtype = float)
        
        for col in range(X.shape[1]):
            a[col], b[col] = np.polyfit(reference, tmp_x[:, col], deg = 1)
        
        self.a_ = a
        self.b_ = b
       
        return self

    def transform(self, X):
        check_is_fitted(self)
        
        X = self._validate_data(
            X, 
            reset = False, 
            copy = self.copy, 
            dtype = FLOAT_DTYPES, 
            estimator = self
        )
        
        if X.shape[1] != len(self.a_) or X.shape[1] != len(self.b_):
            raise ValueError("Transform cannot be applied with provided X. Bad number of columns.")
        
        if self.scale:
            X = self.scaler_.transform(X)
            
        for col in range(X.shape[1]):
            a = self.a_[col]
            b = self.b_[col]
            X[:, col] = (X[:, col] - b) / a
        
        return X
    
    def inverse_transform(self, X):
        check_is_fitted(self)

        X = check_array(
            X, 
            copy = self.copy, 
            dtype = FLOAT_DTYPES
        )

        if X.shape[1] != len(self.a_) or X.shape[1] != len(self.b_):
            raise ValueError("Inverse transform cannot be applied with provided X. Bad number of columns.")
        
        for col in range(X.shape[1]):
            a = self.a_[col]
            b = self.b_[col]
            X[:, col] = (X[:, col] * a) + b
        
        if self.scale:
            X = self.scaler_.inverse_transform(X)
        return X
        
    def _more_tags(self):
        return {'allow_nan': False}




def msc(spectra, scaled = True):
    """ Performs multiplicative scatter correction to the mean.
    Args:
        spectra < numpy.ndarray > : NIRS data matrix.
    Returns:
        spectra < numpy.ndarray > : Scatter corrected NIR spectra.
    """
    if scaled:
        spectra = scale(spectra, with_std = False, axis = 0) # StandardScaler / demean
        
    reference = np.mean(spectra, axis = 1)

    for col in range(spectra.shape[1]):
        a, b = np.polyfit(reference, spectra[:, col], deg = 1)
        spectra[:, col] = (spectra[:, col] - b) / a

    return spectra




def derivate(spectra, order = 1, delta = 1):
    """ Computes Nth order derivates with the desired spacing using numpy.gradient.
    Args:
        spectra < numpy.ndarray > : NIRS data matrix.
        order < float > : Order of the derivation.
        delta < int > : Delta of the derivate (in samples).
    Returns:
        spectra < numpy.ndarray > : Derivated NIR spectra.
    """
    for n in range(order):
        spectra = np.gradient(spectra, delta, axis = 0)
    return spectra


def _gaussian(spectra, order = 1, sigma = 2):
    """ Computes 1D gaussian filter using scipy.ndimage gaussian 1d filter.
    Args:
        spectra < numpy.ndarray > : NIRS data matrix.
        order < float > : Order of the derivation.
        sigma < int > : Sigma of the gaussian.
    Returns:
        spectra < numpy.ndarray > : Gaussian NIR spectra.
    """
    return nd.gaussian_filter1d(spectra, order = order, sigma = sigma)

def gaussian_0(spectra):
    return _gaussian(spectra, order = 0, sigma = 3)

def gaussian_1(spectra):
    return _gaussian(spectra, order = 1, sigma = 2)
    
def gaussian_2(spectra):
    return _gaussian(spectra, order = 2, sigma = 1)


# mode: ['zero', 'constant', 'symmetric', 'periodic', 'smooth', 'periodization', 'reflect', 'antisymmetric', 'antireflect']
# wavelet: ['haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus', 'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor']
def wavelet_transform(spectra, wavelet, mode = "per"):
    """ Computes transform using pywavelet transform.
    Args:
        spectra < numpy.ndarray > : NIRS data matrix.
        wavelet < str > : wavelet family transformation
        mode < str > : signal extension mode
    Returns:
        spectra < numpy.ndarray > : wavelet and resampled spectra.
    """
    _, wt_coeffs = pywt.dwt(spectra, wavelet = wavelet, mode = mode)
    if len(wt_coeffs[0]) != len(spectra[0]):
        return signal.resample(wt_coeffs, len(spectra[0]), axis = 1)
    else:
        return wt_coeffs
    
    
def wv_haar(spectra):
    """ Computes haar transform using pywavelet transform. Spectra is resampled to fit size.
    Args:
        spectra < numpy.ndarray > : NIRS data matrix.
    Returns:
        spectra < numpy.ndarray > : haar transformed spectra.
    """
    return wavelet_transform(spectra, 'haar', 'per')

def spl_norml(spectra):
    """ Perform simple spectral normalisation. (manual algo)
    Args:
        spectra < numpy.ndarray > : NIRS data matrix.
    Returns:
        spectra < numpy.ndarray > : Normalized NIR spectra
    """
    return (spectra - np.min(spectra)) / (np.max(spectra) - np.min(spectra))

# METHOD_DICT = {
#     "baseline": baseline, 
#     "snv": snv, 
#     "rnv": rnv, 
#     "lsnv": lsnv, 
#     "savgol": _savgol, 
#     "savgol0": savgol_0,  
#     "savgol1": savgol_1, 
#     "norml": norml, 
#     "detrend": detrend, 
#     "msc": msc, 
#     "emsc": emsc, 
#     "smooth": smooth, 
#     "derivate": derivate, 
#     "gaussian": _gaussian, 
#     "gaussian_0": gaussian_0, 
#     "gaussian1": gaussian_1, 
#     "gaussian2": gaussian_2, 
#     "wv_haar": wv_haar, 
#     "spl_norml": spl_norml, 
# }


# def process(spectra, processing):
#     pp_spectra = {}
#     pp_spectra['x'] = spectra
#     q = queue.Queue()

#     def processor():
#         while True:
#             item = q.get()
#             if item == 'x':
#                 q.task_done()
#                 continue
#             pipeline = item.split('*')
#             x = pp_spectra['x']
#             state = 0
            
#             # get best existing spectra
#             for i in range(len(pipeline)):
#                 k = '*'.join(pipeline[:-i]) 
#                 if k in pp_spectra:
#                     x = pp_spectra[k]
#                     state = len(pipeline) - i
#                     break
            
#             # apply remaining methods
#             for i in range(state, len(pipeline)):
#                 proc = pipeline[i]
#                 if 'wv_' in proc:
#                     proc = proc.split('_')[1]
#                     x = wavelet_transform(x, proc)
#                     x = RobustScaler().fit_transform(x)
#                     x = spl_norml(x)
#                     pp_spectra[item] = x
#                 else:
#                     x = METHOD_DICT[proc](x)
#                     x = spl_norml(x)
#                     pp_spectra[item] = x
#             q.task_done()
            
#     threading.Thread(target = processor, daemon = True).start()
    
#     for p in processing:
#         q.put(p)
#     q.join()
    
#     return pp_spectra
    
    
# def trim(wavelength, spectra, bins):
#     """ Trim spectra to a specified wavelength bin (or bins).
#     Args:
#         wavelength < numpy.ndarray > : Vector of wavelengths.
#         spectra < numpy.ndarray > : NIRS data matrix.
#         bins < list > : A bin or a list of bins defining the trim operation.
#     Returns:
#         spectra < numpy.ndarray > : NIRS data smoothed with Savitzky-Golay filtering
#     """
#     if type(bins[0]) != list:
#         bins = [bins]

#     spectra_trim = np.array([]).reshape(0, spectra.shape[1])
#     wavelength_trim = np.array([])
#     for wave_range in bins:
#         mask = np.bitwise_and(wavelength >= wave_range[0], wavelength <= wave_range[1])
#         spectra_trim = np.vstack((spectra_trim, spectra[mask, :]))
#         wavelength_trim = np.hstack((wavelength_trim, wavelength[mask]))
#     return wavelength_trim, spectra_trim




# WV_LIST = ['bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8', 'coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'coif6', 'coif7', 'coif8', 'coif9', 'coif10', 'coif11', 'coif12', 'coif13', 'coif14', 'coif15', 'coif16', 'coif17', 'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11', 'db12', 'db13', 'db14', 'db15', 'db16', 'db17', 'db18', 'db19', 'db20', 'db21', 'db22', 'db23', 'db24', 'db25', 'db26', 'db27', 'db28', 'db29', 'db30', 'db31', 'db32', 'db33', 'db34', 'db35', 'db36', 'db37', 'db38', 'dmey', 'haar', 'rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio2.4', 'rbio2.6', 'rbio2.8', 'rbio3.1', 'rbio3.3', 'rbio3.5', 'rbio3.7', 'rbio3.9', 'rbio4.4', 'rbio5.5', 'rbio6.8', 'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', 'sym9', 'sym10', 'sym11', 'sym12', 'sym13', 'sym14', 'sym15', 'sym16', 'sym17', 'sym18', 'sym19', 'sym20']            ]





# class ExtendedMultiplicativeScatterCorrection(TransformerMixin, BaseEstimator):
    
#     def __init__(self, wave, scale = False, *, copy = True):
#         self.copy = copy
#         self.wave = wave

#     def _reset(self):
#         if hasattr(self, "scaler_"):
#             del self.scaler_
#             del self.a_
#             del self.b_
     
#     def fit(self, X, y = None):
#         self._reset()        
#         return self.partial_fit(X, y)
    
#     def partial_fit(self, X, y = None):
        
#         if sparse.issparse(X):
#             raise TypeError("Normalization does not support sparse input")
        
#         first_pass = not hasattr(self, "mean_")
#         X = self._validate_data(
#             X, 
#             reset = first_pass, 
#             dtype = FLOAT_DTYPES, 
#             estimator = self
#         )
        
#         scaler = StandardScaler()
#         scaler.fit(X)
#         self.scaler_ = scaler
        
#         tmp_x = scaler.transform(X, with_std = False, axis = 0)
#         reference = np.mean(tmp_x, axis = 1)
        
#         a = np.empty(X.shape[1], dtype = float)
#         b = np.empty(X.shape[1], dtype = float)
        
#         for col in range(X.shape[1]):
#             a[col], b[col] = np.polyfit(reference, tmp_x[:, col], deg = 1)
        
#         self.a_ = a
#         self.b_ = b
       
#         return self

#     def transform(self, X):
#         check_is_fitted(self)
        
#         X = self._validate_data(
#             X, 
#             reset = False, 
#             copy = self.copy, 
#             dtype = FLOAT_DTYPES, 
#             estimator = self
#         )
        
#         if X.shape[1] != len(self.a_) or X.shape[1] != len(self.b_):
#             raise ValueError("Transform cannot be applied with provided X. Bad number of columns.")
        
#         X = self.scaler_.transform(X)
#         for col in range(X.shape[1]):
#             a = self.a_[col]
#             b = self.b_[col]
#             X[:, col] = (X[:, col] - b) / a
        
#         return X
    
#     def inverse_transform(self, X):
#         check_is_fitted(self)

#         X = check_array(
#             X, 
#             copy = self.copy, 
#             dtype = FLOAT_DTYPES
#         )

#         if X.shape[1] != len(self.a_) or X.shape[1] != len(self.b_):
#             raise ValueError("Inverse transform cannot be applied with provided X. Bad number of columns.")
        
#         for col in range(X.shape[1]):
#             a = self.a_[col]
#             b = self.b_[col]
#             X[:, col] = (X[:, col] * a) + b
        
#         X = self.scaler_.inverse_transform(X)
#         return X
        
#     def _more_tags(self):
#         return {'allow_nan': False}



# def emsc(spectra, wave = [1, 2, 1, 1], scaled = False):
#     """ Performs (basic) extended multiplicative scatter correction to the mean.
#     Args:
#         spectra < numpy.ndarray > : NIRS data matrix.
#     Returns:
#         spectra < numpy.ndarray > : Scatter corrected NIR spectra.
#     """

#     if scaled:
#         spectra = scale(spectra, with_std = False, axis = 0)

#     p1 = .5 * (wave[0] + wave[-1])
#     p2 = 2 / (wave[0] - wave[-1])

#     # Compute model terms
#     model = np.ones((wave.size, 4))
#     model[:, 1] = p2 * (wave[0] - wave) - 1
#     model[:, 2] = (p2 ** 2) * ((wave - p1) ** 2)
#     model[:, 3] = np.mean(spectra, axis = 1)

#     # Solve correction parameters
#     params = np.linalg.lstsq(model, spectra)[0].T

#     # Apply correction
#     spectra = spectra - np.dot(params[:, :-1], model[:, :-1].T).T
#     spectra = np.multiply(spectra, 1 / np.repeat(params[:, -1].reshape(1, -1), spectra.shape[0], axis = 0))

#     return spectra


## @TODO Add convolution
# def smooth(spectra, filter_win = 9, window_type = 'flat', mode = 'reflect'):
#     """ Smooths the spectra using convolution.
#     Args:
#         spectra < numpy.ndarray > : NIRS data matrix.
#         filter_win < float > : length of the filter window in samples.
#         window_type < str > : filtering window to use for convolution (see scipy.signal.windows)
#         mode < str > : convolution mode
#     Returns:
#         spectra < numpy.ndarray > : Smoothed NIR spectra.
#     """

#     if window_type == 'flat':
#         window = np.ones(filter_win)
#     else:
#         window = scipy.signal.windows.get_window(window_type, filter_win)
#     window = window / np.sum(window)

#     for column in range(spectra.shape[1]):
#         spectra[:, column] = nd.convolve(spectra[:, column], window, mode = mode)

#     return spectra

# ### @TODO check if function is useful for our purpose (I think not)
# def lsnv(spectra, num_windows = 10):
#     """ Perform local scatter correction using the standard normal variate.
#     Args:
#         spectra < numpy.ndarray > : NIRS data matrix.
#         num_windows < int > : number of equispaced windows to use (window size (in points) is length / num_windows)
#     Returns:
#         spectra < numpy.ndarray > : NIRS data with local SNV applied.
#     """

#     parts = np.array_split(spectra, num_windows, axis = 0)
#     for idx, part in enumerate(parts):
#         parts[idx] = snv(part)

#     return np.concatenate(parts, axis = 0)
# ####






# class StandardNormalVariate(TransformerMixin, BaseEstimator):
    
#     def __init__(self, *, copy = True):
#         self.copy = copy

#     def _reset(self):
#         if hasattr(self, "mean_"):
#             del self.mean_
#             del self.std_

#     def fit(self, X, y = None):
#         self._reset()        
#         return self.partial_fit(X, y)
    
#     def partial_fit(self, X, y = None):
#         if sparse.issparse(X):
#             raise TypeError("Standard Normal Variate does not support sparse input")
        
#         first_pass = not hasattr(self, "mean_")
#         X = self._validate_data(
#             X, 
#             reset = first_pass, 
#             dtype = FLOAT_DTYPES, 
#             estimator = self
#         )
        
#         self.mean_ = np.mean(X, axis = 0)
#         self.std_ = np.std(X, axis = 0)
#         return self

#     def transform(self, X):
#         check_is_fitted(self)
        
#         X = self._validate_data(
#             X, 
#             reset = False, 
#             copy = self.copy, 
#             dtype = FLOAT_DTYPES, 
#             estimator = self
#         )
        
#         X = (X - self.mean_) / self.std_
#         return X
    
#     def inverse_transform(self, X):
#         check_is_fitted(self)

#         X = check_array(
#             X, 
#             copy = self.copy, 
#             dtype = FLOAT_DTYPES
#         )

#         X = (X * self.std_) + self.mean_
#         return X

#     def _more_tags(self):
#         return {'allow_nan': False}
    
    
# def snv(spectra):
#     """ Perform scatter correction using the standard normal variate.
#     Args:
#         spectra < numpy.ndarray > : NIRS data matrix.
#     Returns:
#         spectra < numpy.ndarray > : NIRS data with (S/R)NV applied.
#     """

#     return (spectra - np.mean(spectra, axis = 1)) / np.std(spectra, axis = 1)


# class RobustNormalVariate(TransformerMixin, BaseEstimator):

#     def __init__(self, iqr = [75, 25], *, copy = True):
#         self.copy = copy
#         self.iqr = iqr

#     def _reset(self):
#         if hasattr(self, "median_"):
#             del self.median_
#             del self.percentile_

#     def fit(self, X, y = None):
#         self._reset()        
#         return self.partial_fit(X, y)
    
#     def partial_fit(self, X, y = None):
#         iqr = self.iqr
#         if len(iqr) < 2 or iqr[0] <= iqr[1] or all((v < 0) and (v > 100) for v in iqr):
#             raise ValueError(
#                 "Interquartile range is not correctly defined. Got %s."
#                 % str(iqr)
#             )
            
#         if sparse.issparse(X):
#             raise TypeError("Robust Normal Variate does not support sparse input")
        
#         first_pass = not hasattr(self, "median_")
#         X = self._validate_data(
#             X, 
#             reset = first_pass, 
#             dtype = FLOAT_DTYPES, 
#             estimator = self
#         )
        
#         self.median_ = np.median(X, axis = 0)
#         self.percentile_ = np.percentile(X, iqr, axis = 0)
#         return self

#     def transform(self, X):
#         check_is_fitted(self)
        
#         X = self._validate_data(
#             X, 
#             reset = False, 
#             copy = self.copy, 
#             dtype = FLOAT_DTYPES, 
#             estimator = self
#         )
        
#         X = (X - self.median_) / np.subtract(*self.percentile_)
#         return X
    
#     def _more_tags(self):
#         return {'allow_nan': False}
    

# def rnv(spectra, iqr = [75, 25]):
#     """ Perform scatter correction using robust normal variate.
#     Args:
#         spectra < numpy.ndarray > : NIRS data matrix.
#         iqr < list > : IQR ranges [lower, upper] for robust normal variate.
#     Returns:
#         spectra < numpy.ndarray > : NIRS data with (S/R)NV applied.
#     """

#     return (spectra - np.median(spectra, axis = 1)) / np.subtract(*np.percentile(spectra, iqr, axis = 1))
