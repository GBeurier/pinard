import warnings
import numpy as np

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


# class Generator(TransformerMixin, BaseEstimator):
    
#     def __init__(self, count = 1, *, copy = True):
#         self.copy = copy


#     def fit(self, X, y = None):
#         self._reset()        
#         return self.partial_fit(X, y)
    
#     def partial_fit(self, X, y = None):
#         if sparse.issparse(X):
#             raise TypeError("Baseline does not support sparse input")
        
#         first_pass = not hasattr(self, "mean_")
#         X = self._validate_data(
#             X, 
#             reset = first_pass, 
#             dtype = FLOAT_DTYPES, 
#             estimator = self
#         )
        
#         self.mean_ = np.mean(X, axis = 0)
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
        
#         X = X - self.mean_
#         return X
    
#     def inverse_transform(self, X):
#         check_is_fitted(self)

#         X = check_array(
#             X, 
#             copy = self.copy, 
#             dtype = FLOAT_DTYPES
#         )

#         X = X + self.mean_
#         return X
    
#     def _more_tags(self):
#         return {'allow_nan': False}