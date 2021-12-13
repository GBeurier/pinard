import numpy as np
import scipy.signal as signal
import scipy.ndimage as nd
from sklearn.preprocessing import scale
import pywt
import threading, queue
from sklearn.preprocessing import RobustScaler

def baseline(spectra):
    """ Removes baseline (mean) from each spectrum.
    Args:
        spectra < numpy.ndarray > : NIRS data matrix.
    Returns:
        spectra < numpy.ndarray > : Mean-centered NIRS data matrix
    """

    return spectra - np.mean(spectra, axis = 0)


def snv(spectra):
    """ Perform scatter correction using the standard normal variate.
    Args:
        spectra < numpy.ndarray > : NIRS data matrix.
    Returns:
        spectra < numpy.ndarray > : NIRS data with (S/R)NV applied.
    """

    return (spectra - np.mean(spectra, axis = 0)) / np.std(spectra, axis = 0)


def rnv(spectra, iqr = [75, 25]):
    """ Perform scatter correction using robust normal variate.
    Args:
        spectra < numpy.ndarray > : NIRS data matrix.
        iqr < list > : IQR ranges [lower, upper] for robust normal variate.
    Returns:
        spectra < numpy.ndarray > : NIRS data with (S/R)NV applied.
    """

    return (spectra - np.median(spectra, axis = 0)) / np.subtract(*np.percentile(spectra, iqr, axis = 0))


def lsnv(spectra, num_windows = 10): ### USELESS FOR NOW
    """ Perform local scatter correction using the standard normal variate.
    Args:
        spectra < numpy.ndarray > : NIRS data matrix.
        num_windows < int > : number of equispaced windows to use (window size (in points) is length / num_windows)
    Returns:
        spectra < numpy.ndarray > : NIRS data with local SNV applied.
    """

    parts = np.array_split(spectra, num_windows, axis = 0)
    for idx, part in enumerate(parts):
        parts[idx] = snv(part)

    return np.concatenate(parts, axis = 0)


def _savgol(spectra, filter_win = 11, poly_order = 3, deriv_order = 0, delta = 1.0):
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

def savgol_0(spectra):
    return _savgol(spectra, 11, 3, 0)

def savgol_1(spectra):
    return _savgol(spectra, 11, 3, 1)

def norml(spectra, udefined = True, imin = 0, imax = 1):
    """ Perform spectral normalisation with user-defined limits. (numpy linalg)
    Args:
        spectra < numpy.ndarray > : NIRS data matrix.
        udefined < bool > : use user defined limits
        imin < float > : user defined minimum
        imax < float > : user defined maximum
    Returns:
        spectra < numpy.ndarray > : Normalized NIR spectra
    """
    if udefined:
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


def msc(spectra):
    """ Performs multiplicative scatter correction to the mean.
    Args:
        spectra < numpy.ndarray > : NIRS data matrix.
    Returns:
        spectra < numpy.ndarray > : Scatter corrected NIR spectra.
    """

    spectra = scale(spectra, with_std = False, axis = 0) # Demean
    reference = np.mean(spectra, axis = 1)

    for col in range(spectra.shape[1]):
        a, b = np.polyfit(reference, spectra[:, col], deg = 1)
        spectra[:, col] = (spectra[:, col] - b) / a

    return spectra


def emsc(spectra, wave = [1, 2, 1, 1], remove_mean = False):
    """ Performs (basic) extended multiplicative scatter correction to the mean.
    Args:
        spectra < numpy.ndarray > : NIRS data matrix.
    Returns:
        spectra < numpy.ndarray > : Scatter corrected NIR spectra.
    """

    if remove_mean:
        spectra = scale(spectra, with_std = False, axis = 0)

    p1 = .5 * (wave[0] + wave[-1])
    p2 = 2 / (wave[0] - wave[-1])

    # Compute model terms
    model = np.ones((wave.size, 4))
    model[:, 1] = p2 * (wave[0] - wave) - 1
    model[:, 2] = (p2 ** 2) * ((wave - p1) ** 2)
    model[:, 3] = np.mean(spectra, axis = 1)

    # Solve correction parameters
    params = np.linalg.lstsq(model, spectra)[0].T

    # Apply correction
    spectra = spectra - np.dot(params[:, :-1], model[:, :-1].T).T
    spectra = np.multiply(spectra, 1 / np.repeat(params[:, -1].reshape(1, -1), spectra.shape[0], axis = 0))

    return spectra



def smooth(spectra, filter_win = 9, window_type = 'flat', mode = 'reflect'):
    """ Smooths the spectra using convolution.
    Args:
        spectra < numpy.ndarray > : NIRS data matrix.
        filter_win < float > : length of the filter window in samples.
        window_type < str > : filtering window to use for convolution (see scipy.signal.windows)
        mode < str > : convolution mode
    Returns:
        spectra < numpy.ndarray > : Smoothed NIR spectra.
    """

    if window_type == 'flat':
        window = np.ones(filter_win)
    else:
        window = scipy.signal.windows.get_window(window_type, filter_win)
    window = window / np.sum(window)

    for column in range(spectra.shape[1]):
        spectra[:, column] = nd.convolve(spectra[:, column], window, mode = mode)

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

METHOD_DICT = {
    "baseline": baseline,
    "snv": snv, 
    "rnv": rnv, 
    "lsnv": lsnv,
    "savgol": _savgol, 
    "savgol0": savgol_0,  
    "savgol1": savgol_1, 
    "norml": norml, 
    "detrend": detrend,
    "msc": msc, 
    "emsc": emsc, 
    "smooth": smooth, 
    "derivate": derivate, 
    "gaussian": _gaussian, 
    "gaussian_0": gaussian_0, 
    "gaussian1": gaussian_1, 
    "gaussian2": gaussian_2, 
    "wv_haar": wv_haar,
    "spl_norml": spl_norml,
}


def process(spectra, processing):
    pp_spectra = {}
    pp_spectra['x'] = spectra
    q = queue.Queue()

    def processor():
        while True:
            item = q.get()
            if item == 'x':
                q.task_done()
                continue
            pipeline = item.split('*')
            x = pp_spectra['x']
            state = 0
            
            # get best existing spectra
            for i in range(len(pipeline)):
                k = '*'.join(pipeline[:-i]) 
                if k in pp_spectra:
                    x = pp_spectra[k]
                    state = len(pipeline) - i
                    break
            
            # apply remaining methods
            for i in range(state, len(pipeline)):
                proc = pipeline[i]
                if 'wv_' in proc:
                    proc = proc.split('_')[1]
                    x = wavelet_transform(x, proc)
                    x = RobustScaler().fit_transform(x)
                    x = spl_norml(x)
                    pp_spectra[item] = x
                else:
                    x = METHOD_DICT[proc](x)
                    x = spl_norml(x)
                    pp_spectra[item] = x
            q.task_done()
            
    threading.Thread(target=processor, daemon=True).start()
    
    for p in processing:
        q.put(p)
    q.join()
    
    return pp_spectra
    
    
def trim(wavelength, spectra, bins):
    """ Trim spectra to a specified wavelength bin (or bins).
    Args:
        wavelength < numpy.ndarray > : Vector of wavelengths.
        spectra < numpy.ndarray > : NIRS data matrix.
        bins < list > : A bin or a list of bins defining the trim operation.
    Returns:
        spectra < numpy.ndarray > : NIRS data smoothed with Savitzky-Golay filtering
    """
    if type(bins[0]) != list:
        bins = [bins]

    spectra_trim = np.array([]).reshape(0, spectra.shape[1])
    wavelength_trim = np.array([])
    for wave_range in bins:
        mask = np.bitwise_and(wavelength >= wave_range[0], wavelength <= wave_range[1])
        spectra_trim = np.vstack((spectra_trim, spectra[mask, :]))
        wavelength_trim = np.hstack((wavelength_trim, wavelength[mask]))
    return wavelength_trim, spectra_trim




# WV_LIST = ['bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8', 'coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'coif6', 'coif7', 'coif8', 'coif9', 'coif10', 'coif11', 'coif12', 'coif13', 'coif14', 'coif15', 'coif16', 'coif17', 'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11', 'db12', 'db13', 'db14', 'db15', 'db16', 'db17', 'db18', 'db19', 'db20', 'db21', 'db22', 'db23', 'db24', 'db25', 'db26', 'db27', 'db28', 'db29', 'db30', 'db31', 'db32', 'db33', 'db34', 'db35', 'db36', 'db37', 'db38', 'dmey', 'haar', 'rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio2.4', 'rbio2.6', 'rbio2.8', 'rbio3.1', 'rbio3.3', 'rbio3.5', 'rbio3.7', 'rbio3.9', 'rbio4.4', 'rbio5.5', 'rbio6.8', 'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', 'sym9', 'sym10', 'sym11', 'sym12', 'sym13', 'sym14', 'sym15', 'sym16', 'sym17', 'sym18', 'sym19', 'sym20']            ]

