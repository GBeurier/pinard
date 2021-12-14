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
from tensorflow.keras.layers import Dense, Conv1D, Flatten, BatchNormalization, SpatialDropout1D, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from kennard_stone import train_test_split, KFold
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

hello = tf.constant("hello TensorFlow!")

def load_path(folder):
    n = NSet("example")
    n.load(folder+"/XCal.csv", folder+"/YCal.csv", 0, 0, 0)
    x_src = n.get_raw_x()

    x = x_src.copy()
    # PROCESSING = ['x','x*detrend','x*snv','x*savgol1','x*msc','x*derivate','x*gaussian1','x*gaussian2','x*wv_haar','x*detrend*snv','x*detrend*rnv','x*detrend*savgol1','x*detrend*msc','x*detrend*derivate','x*detrend*wv_haar','x*wv_bior2.2','x*wv_db2','x*wv_dmey','x*wv_rbio6.8','x*wv_sym4','x*snv*savgol1','x*savgol1*savgol1','x*msc*savgol1','x*derivate*savgol1','x*gaussian1*savgol1','x*gaussian2*savgol1','x*wv_haar*savgol1','x*snv*msc','x*rnv*msc','x*savgol1*msc','x*msc*msc','x*derivate*msc','x*gaussian1*msc','x*gaussian2*msc','x*wv_haar*msc','x*snv*derivate','x*savgol1*derivate','x*msc*derivate','x*derivate*derivate','x*gaussian1*derivate','x*gaussian2*derivate','x*wv_haar*derivate']
    PROCESSING = ['x'
                # ,'x*detrend'
                ,'x*savgol1'
                # ,'x*derivate'
                ,'x*gaussian1'
                ,'x*gaussian2'
                ,'x*wv_haar'
                # ,'x*detrend*savgol1'
                # ,'x*detrend*derivate'
                # ,'x*detrend*wv_haar'
                # ,'x*detrend*gaussian1'
                # ,'x*detrend*gaussian2'
                # ,'x*wv_bior2.2'
                # ,'x*wv_db2'
                # ,'x*wv_dmey'
                # ,'x*wv_rbio6.8'
                # ,'x*wv_sym4'
                ,'x*savgol1*savgol1'
                # ,'x*derivate*savgol1'
                ,'x*gaussian1*savgol1'
                ,'x*gaussian2*savgol1'
                ,'x*wv_haar*savgol1'
                # ,'x*savgol1*derivate'
                # ,'x*derivate*derivate'
                # ,'x*gaussian1*derivate'
                # ,'x*gaussian2*derivate'
                # ,'x*wv_haar*derivate'
                ]
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
    # model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Conv1D (filters=64, kernel_size=21, strides=3, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Conv1D (filters=32, kernel_size=5, strides=3, activation='elu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    # model.summary()

    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, verbose=1, min_delta=0.5e-5, mode='min')
    earlyStopping = EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='min') 
    mcp_save = ModelCheckpoint("tmp.model", save_best_only=True, monitor='val_loss', mode='min') 

    model.compile(loss='mean_squared_error', metrics=['mae','mse'], optimizer='rmsprop')

    history = model.fit(X_train, y_train, 
                epochs=1200, 
                batch_size=1000, 
                shuffle=True, 
                validation_data = (X_test, y_test),
                verbose=0, 
                callbacks=[earlyStopping])

    best_score = min(history.history['val_mse'])
    best_epoch = np.argmin(history.history['val_mse'])
    print(folder, input_shape, name, best_score, best_epoch)
    return best_score
    
    
def benchmark_folder(x,y,folder):
    print("***** BENCHMARKING", folder, "*****")
    s = x.shape
    
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
    
    scores = {}
    
    kf = KFold(n_splits=4)#, random_state=SEED, shuffle=True)
    fold_index = 0
    for train_index, test_index in kf.split(x): #sklearn split
    # for train_index, test_index in kf.split(x[:,:,0]): #kennard stone split
        
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        score = learn(X_train, X_test, y_train, y_test, (s[1], s[2]), fold_index, folder)
        name = str(fold_index)
        scores[name] = score
            
        # X_train_src = X_train[:,:,0:2]
        # X_test_src = X_test[:,:,0:2]
        
        # for i in range(15, s[2], 15):
        #     X_train_sub = X_train[:,:,0:i]
        #     X_test_sub = X_test[:,:,0:i]
        #     score = learn(X_train_sub, X_test_sub, y_train, y_test, (s[1], i), str(i), folder)
        #     name = str(fold_index) + '>0-' + str(i)
        #     scores[name] = score
            
        # for i in range(15, s[2], 15):
        #     X_train_sub = X_train[:,:,i-15:i]
        #     X_train_sub = np.concatenate((X_train_src, X_train_sub), axis = 2)
        #     X_test_sub = X_test[:,:,i-15:i]
        #     X_test_sub = np.concatenate((X_test_src, X_test_sub), axis = 2)
        #     score = learn(X_train_sub, X_test_sub, y_train, y_test, (s[1], 17), str(i), folder)
        #     name = str(fold_index) + '>' + str(i-15) + "-" + str(i)
        #     scores[name] = score
            
        fold_index += 1
        
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