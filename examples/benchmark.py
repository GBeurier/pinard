def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import json
import numpy as np
import sys
import os
import random
sys.path.append(os.path.abspath("") + "/../src/pynirs")
from nirs_set import NIRS_Set as NSet
import preprocessor as pp
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, BatchNormalization, SpatialDropout1D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from kennard_stone import train_test_split
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from matplotlib.pyplot import plot

SEED = 12485
def set_seed(sd):
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED']=str(sd)
    random.seed(sd)
    np.random.seed(sd)
    tf.random.set_seed(sd)
set_seed(SEED)

def load_path(folder):
    n = NSet("example")
    n.load(folder+"/XCal.csv", folder+"/YCal.csv", 0, 0, 0)
    x_src = n.get_raw_x()

    x = x_src.copy()
    PROCESSING = ['x', 'x*detrend', 'x*snv', 'x*rnv', 'x*savgol1', 'x*msc', 'x*derivate', 'x*gaussian1', 'x*gaussian2', 'x*wv_haar', 'x*detrend*snv', 'x*detrend*rnv', 'x*detrend*savgol1', 'x*detrend*msc', 'x*detrend*derivate', 'x*detrend*gaussian1', 'x*detrend*gaussian2', 'x*detrend*wv_haar', 'x*wv_bior2.2', 'x*wv_bior3.1', 'x*wv_bior4.4', 'x*wv_bior5.5', 'x*wv_bior6.8', 'x*wv_coif1', 'x*wv_coif2', 'x*wv_coif3', 'x*wv_coif4', 'x*wv_coif5', 'x*wv_coif6', 'x*wv_coif7', 'x*wv_coif8', 'x*wv_coif9', 'x*wv_coif10', 'x*wv_coif11', 'x*wv_coif12', 'x*wv_coif13', 'x*wv_coif14', 'x*wv_coif15', 'x*wv_coif16', 'x*wv_coif17', 'x*wv_db2', 'x*wv_db3', 'x*wv_db4', 'x*wv_db5', 'x*wv_db6', 'x*wv_db7', 'x*wv_db8', 'x*wv_db9', 'x*wv_db10', 'x*wv_db11', 'x*wv_db12', 'x*wv_db13', 'x*wv_db14', 'x*wv_db15', 'x*wv_db16', 'x*wv_db17', 'x*wv_db18', 'x*wv_db19', 'x*wv_db20', 'x*wv_db21', 'x*wv_db22', 'x*wv_db23', 'x*wv_db24', 'x*wv_db25', 'x*wv_db26', 'x*wv_db27', 'x*wv_db28', 'x*wv_db29', 'x*wv_db30', 'x*wv_db31', 'x*wv_db32', 'x*wv_db33', 'x*wv_db34', 'x*wv_db35', 'x*wv_db36', 'x*wv_db37', 'x*wv_db38', 'x*wv_dmey', 'x*wv_rbio2.2', 'x*wv_rbio3.1', 'x*wv_rbio4.4', 'x*wv_rbio5.5', 'x*wv_rbio6.8', 'x*wv_sym4', 'x*wv_sym5', 'x*wv_sym6', 'x*wv_sym7', 'x*wv_sym8', 'x*wv_sym9', 'x*wv_sym10', 'x*wv_sym11', 'x*wv_sym12', 'x*wv_sym13', 'x*wv_sym14', 'x*wv_sym15', 'x*wv_sym16', 'x*wv_sym17', 'x*wv_sym18', 'x*wv_sym19', 'x*wv_sym20', 'x*snv*snv', 'x*rnv*snv', 'x*savgol1*snv', 'x*msc*snv', 'x*derivate*snv', 'x*gaussian1*snv', 'x*gaussian2*snv', 'x*wv_haar*snv', 'x*snv*rnv', 'x*rnv*rnv', 'x*savgol1*rnv', 'x*msc*rnv', 'x*derivate*rnv', 'x*gaussian1*rnv', 'x*gaussian2*rnv', 'x*wv_haar*rnv', 'x*snv*savgol1', 'x*rnv*savgol1', 'x*savgol1*savgol1', 'x*msc*savgol1', 'x*derivate*savgol1', 'x*gaussian1*savgol1', 'x*gaussian2*savgol1', 'x*wv_haar*savgol1', 'x*snv*msc', 'x*rnv*msc', 'x*savgol1*msc', 'x*msc*msc', 'x*derivate*msc', 'x*gaussian1*msc', 'x*gaussian2*msc', 'x*wv_haar*msc', 'x*snv*derivate', 'x*rnv*derivate', 'x*savgol1*derivate', 'x*msc*derivate', 'x*derivate*derivate', 'x*gaussian1*derivate', 'x*gaussian2*derivate', 'x*wv_haar*derivate', 'x*snv*gaussian1', 'x*rnv*gaussian1', 'x*savgol1*gaussian1', 'x*msc*gaussian1', 'x*derivate*gaussian1', 'x*gaussian1*gaussian1', 'x*gaussian2*gaussian1', 'x*wv_haar*gaussian1', 'x*snv*gaussian2', 'x*rnv*gaussian2', 'x*savgol1*gaussian2', 'x*msc*gaussian2', 'x*derivate*gaussian2', 'x*gaussian1*gaussian2', 'x*gaussian2*gaussian2', 'x*wv_haar*gaussian2', 'x*snv*wv_haar', 'x*rnv*wv_haar', 'x*savgol1*wv_haar', 'x*msc*wv_haar', 'x*derivate*wv_haar', 'x*gaussian1*wv_haar', 'x*gaussian2*wv_haar', 'x*wv_haar*wv_haar']
    pp_spectra = pp.process(x, PROCESSING)
    x = np.array(list(pp_spectra.values()))
    x = np.swapaxes(x, 0, 1)
    x = np.swapaxes(x, 1, 2)
    
    y_src = n.get_raw_y()
    scaler_y = MinMaxScaler(feature_range=(0.1,0.9))
    y = scaler_y.fit_transform(y_src)
    
    return x, y, scaler_y

def learn(X_train, X_test, y_train, y_test, input_shape, name, folder):
    model = Sequential()
    model.add(SpatialDropout1D(0.08, input_shape=input_shape))
    model.add(Conv1D (filters=8, kernel_size=15, strides=5, activation='selu'))
    model.add(BatchNormalization())
    model.add(Conv1D (filters=64, kernel_size=21, strides=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv1D (filters=32, kernel_size=5, strides=3, activation='elu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    # model.summary()

    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, verbose=1, min_delta=0.5e-5, mode='min')
    earlyStopping = EarlyStopping(monitor='val_loss', patience=150, verbose=0, mode='min') 
    mcp_save = ModelCheckpoint("tmp.model", save_best_only=True, monitor='val_loss', mode='min') 

    model.compile(loss='mean_squared_error', metrics=['mae','mse'], optimizer='rmsprop')

    history = model.fit(X_train, y_train, 
                epochs=250, 
                batch_size=500, 
                shuffle=True, 
                validation_data = (X_test, y_test),
                verbose=0, 
                callbacks=[reduce_lr_loss, earlyStopping])

    best_score = min(history.history['val_mse'])
    print(folder, input_shape, name, best_score)
    return best_score
    
    
def benchmark_folder(x,y,folder):
    print("***** BENCHMARKING", folder, "*****")
    s = x.shape
    
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
    
    scores = {}
    
    kf = KFold(n_splits=5, random_state=SEED, shuffle=True)
    for train_index, test_index in kf.split(x):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        X_train_src = X_train[:,:,0:2]
        X_test_src = X_test[:,:,0:2]
        
        for i in range(15, s[2], 15):
            X_train_sub = X_train[:,:,0:i]
            X_test_sub = X_test[:,:,0:i]
            score = learn(X_train_sub, X_test_sub, y_train, y_test, (s[1], i), str(i), folder)
            name = '0-' + str(i)
            scores[name] = score
            
        for i in range(15, s[2], 15):
            X_train_sub = X_train[:,:,i-15:i]
            X_train_sub = np.concatenate((X_train_src, X_train_sub), axis = 2)
            X_test_sub = X_test[:,:,i-15:i]
            X_test_sub = np.concatenate((X_test_src, X_test_sub), axis = 2)
            score = learn(X_train_sub, X_test_sub, y_train, y_test, (s[1], 17), str(i), folder)
            name = str(i-15) + "-" + str(i)
            scores[name] = score
            
    return scores
    
    
def traverse(directory):
    scores = {}
    folders = [x[0] for x in os.walk(directory)]
    for f in folders:
        if f == directory:
            continue
        print("LOADING", f)
        x,y,_ = load_path(f)
        score = benchmark_folder(x,y,f)
        scores[f] = score
    return scores

scores = traverse('sample_data')
with open("scores.json", "w") as write_file:
    json.dump(scores, write_file, indent=4)