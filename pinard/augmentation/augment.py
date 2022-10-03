import warnings
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
import random ## TODO replace by np.random
import scipy.interpolate as interpolate

############################
##    HELPERS FUNCTIONS   ##
############################

def anglePointFixe(x,xI,yI,p1,p2):   
    if x <= xI:
        return(p1*(x-xI)+yI)
    else:
        return(p2*(x-xI)+yI)

v_anglePointFixe = np.vectorize(anglePointFixe)


def segment_length(x1,y1,x2,y2):
    return(np.sqrt((x2-x1)**2+(y2-y1)**2))

v_segment_length = np.vectorize(segment_length)


def spectrum_length(x,y):
    x1 = x[range((len(x)-1))]
    y1 = y[range((len(y)-1))]
    x2 = x[range(1,len(x))]
    y2 = y[range(1,len(y))]
    SpecLen_seg = v_segment_length(x1,y1,x2,y2)
    SpecLen = np.sum(SpecLen_seg)
    SpecLen_seg_cumsum = np.cumsum(SpecLen_seg)
    return(SpecLen,SpecLen_seg,SpecLen_seg_cumsum) 
    
def segment_pt_coord(x1,y1,x2,y2,fracL,L):
    propL = fracL/L
    xp = x1 + propL*(x2-x1)
    yp = y1 + propL*(y2-y1)
    return(xp,yp)        

def interval_selection(n_l,CumVect):
    i1 = np.where(n_l<=CumVect) 
    i2 = np.where(n_l>=CumVect)
    return(np.min(i1),np.max(i2))
    
def nearest_x(x_test,vx):
    index_min=0
    dist_min = abs(x_test-vx[index_min])
    for i in range(1,len(vx)):
        dist = abs(x_test-vx[i])
        if dist < dist_min: 
            dist_min=dist
            index_min=i
    return(index_min)        


#####################
##    GENERATORS   ##
#####################
    # def fit_transform(self, X, y=None, **fit_params):
    #     return super().fit_transform(X, y)
    
    # def fit(self, X, y = None):
    #     return super().fit(X, y)

    # def transform(self, X, y):
    #     return super().transform(X,y)
        


class Augmenter(TransformerMixin, BaseEstimator):
    
    def __init__(self, count = 1):
        self.count = count

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X, y)
    
    def fit(self, X, y = None):
        return self

    def augment(self, X, y):
        new_X = X.copy()
        new_y = y.copy()
        return new_X, new_y

    def transform(self, X, y):
        self.init_size = len(X)
        for i in range(len(X)):
            for k in range(self.count):
                newX, newy = self.augment(X[i], y[i])
                if isinstance(newX, np.ndarray):
                    X = np.vstack([X, newX])
                    y = np.vstack([y, newy])
        return X, y
    
    def inverse_transform(self, X, y):
        return X[0:self.init_size], y[0:self.init_size]
    
    def _more_tags(self):
        return {'allow_nan': False}
    
    
    
class SampleAugmentation(Augmenter):
    def augment(self, X, y):
        return None, None
    
class Rotate_Translate(Augmenter):
    def augment(self, spectrum, y):
        """ rotate and translate signal """
        x_range = np.linspace(0, 1, len(spectrum))
        p2 = random.uniform(-2,2)
        p1 = random.uniform(-2,2)
        xI = random.uniform(0,1)
        yI = random.uniform(0,np.max(spectrum)/3)
        distor = v_anglePointFixe(x_range,xI,yI,p1,p2)
        distor = distor*np.std(spectrum)
        y_distor = spectrum + distor
        # y_distor[y_distor < 0] = 0
        return np.array(y_distor), y
    
    

class Monotonous_Spline_Simplification(Augmenter):
    def augment(self, spectrum, y):
        """ Select regularly spaced points along x_axis and adjust a spline """
        nfreq = len(spectrum)
        x_range_real = np.arange(0, nfreq, 1)
        nb_points = 60

        ctrl_points = np.unique(np.concatenate(([0],random.sample(range(nfreq),nb_points),[nfreq-1])))
        ctrl_points.sort()
        x = x_range_real[ctrl_points]
        y = spectrum[ctrl_points]
        t, c, k = interpolate.splrep(x, y, s=0, k=3)
        spline = interpolate.BSpline(t, c, k, extrapolate=False)
        
        return spline(x_range_real), y


class Dependent_Spline_Simplification(Augmenter):
    def augment(self, spectrum, y):
        """ Select regularly spaced points ON the spectrum and adjust a spline """
        nfreq = len(spectrum)
        x0 = np.linspace(0, np.max(spectrum), nfreq)
        res = spectrum_length(x0,spectrum)
        nb_segments = 10
        x_samples = []
        y_samples = []

        for s in range(1,nb_segments):
            l = spectrum_length(x0,spectrum)[0] / nb_segments
            # cumulative_length = np.cumsum(np.repeat(l,nb_segments))
            n_l = s*l
            test = res[2]
            toto = interval_selection(n_l,test)
            P = segment_pt_coord(
                x1=x0[toto[1]],
                y1=spectrum[toto[1]],
                x2=x0[toto[0]],
                y2=spectrum[toto[0]],
                fracL=res[1][toto[1]]%l,
                L=res[1][toto[1]]) 

            x_samples.append(P[0])
            y_samples.append(P[1])
        
        x = np.array(x_samples)
        x = np.concatenate(([0],x,[np.max(x0)]))
        y = np.array(y_samples)
        y = np.concatenate(([spectrum[0]],y,[spectrum[nfreq-1]]))
        # print(x)
        t, c, k = interpolate.splrep(x, y, s=0, k=3)
        xmin, xmax = x.min(), x.max()
        xx = np.linspace(xmin, xmax, nfreq)
        spline = interpolate.BSpline(t, c, k, extrapolate=False)

        return spline(xx), y

class Random_Spline_Addition(Augmenter):
    def augment(self, spectrum, y):
        """ Add spline noise on y """
        nfreq = len(spectrum)
        x_range_real = np.arange(0, nfreq, 1)
        interval_width = np.max(spectrum)/80
        half_interval_width = interval_width/2 
        baseline = random.uniform(-half_interval_width, half_interval_width)
        interval_min = -half_interval_width + baseline
        interval_max = half_interval_width + baseline

        nb_spline_points = 40
        x_points = np.linspace(0, nfreq,nb_spline_points)
        y_points = [random.uniform(interval_min,interval_max) for i in range(nb_spline_points)]

        x = np.asarray(x_points)
        x.sort()
        y = np.asarray(y_points)
        t, c, k = interpolate.splrep(x, y, s=0, k=3)

        spline = interpolate.BSpline(t, c, k, extrapolate=False)
        distor = spline(x_range_real)
        y_distor = spectrum + distor
        # y_distor[y_distor < 0] = 0
        return np.array(y_distor), y

class Random_Y_Shift(Augmenter):
    def augment(self, spectrum, y):
        """ Additive delta on y """
        spec_range = np.max(spectrum) - np.min(spectrum)
        y_distor = spectrum + random.uniform(-spec_range,spec_range) / 20
        return y_distor, y

class Random_Multiplicative_Shift(Augmenter):
    def augment(self, spectrum, y):
        """ Multiplicative delta on y """
        y_distor = spectrum * random.uniform (-0.05, 0.05)
        return y_distor, y


class Random_X_Spline_Deformation(Augmenter):
    def augment(self, spectrum, y):
        """ Random modification of x based on subsampled spline """
        x = np.arange(0, len(spectrum), 1)
        t, c, k = interpolate.splrep(x, spectrum, s=0, k=3)

        delta_x_size = int(np.around(len(t)/20))
        delta_x = np.linspace(np.min(x), np.max(x), delta_x_size)
        delta_y = np.random.uniform(-10,10,delta_x_size)
        delta = np.interp(t, delta_x, delta_y)
        t = t + delta
        spline = interpolate.BSpline(t, c, k, extrapolate=True)
        return spline(x), y

class Random_X_Spline_Shift(Augmenter):
    def augment(self, spectrum, y):
        """ Add spline based x shift """
        x = np.arange(0, len(spectrum), 1)
        delta = random.uniform(-10,10)
        x = x + delta
        t, c, k = interpolate.splrep(x, spectrum, s=0, k=3)
        spline = interpolate.BSpline(t, c, k, extrapolate=False)
        return spline(x), y

