from tensorflow.keras.layers import Dense, Conv1D, Activation, SpatialDropout1D,BatchNormalization,Flatten, Dropout, \
                                    Input, MaxPool1D, SeparableConv1D, Add, GlobalAveragePooling1D, \
                                    DepthwiseConv1D, MaxPooling1D
from tensorflow.keras.models import Model, Sequential



def xception_entry_flow(inputs) :
    x = Conv1D(32, 3, strides = 2, padding='same')(inputs)
    x = SpatialDropout1D(0.3)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv1D(64,3,padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    previous_block_activation = x

    for size in [128, 256, 728] :

        x = Activation('relu')(x)
        x = SeparableConv1D(size, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = SeparableConv1D(size, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = MaxPool1D(3, strides=2, padding='same')(x)

        residual = Conv1D(size, 1, strides=2, padding='same')(previous_block_activation)

        x = Add()([x, residual])
        previous_block_activation = x

    return x

def xception_middle_flow(x, num_blocks=8) :
    
    previous_block_activation = x

    for _ in range(num_blocks) :

        x = Activation('relu')(x)
        x = SeparableConv1D(728, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = SeparableConv1D(728, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = SeparableConv1D(728, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = Add()([x, previous_block_activation])
        previous_block_activation = x

    return x

def xception_exit_flow(x) :
    
    previous_block_activation = x

    x = Activation('relu')(x)
    x = SeparableConv1D(728, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv1D(1024, 3, padding='same')(x) 
    x = BatchNormalization()(x)

    x = MaxPool1D(3, strides=2, padding='same')(x)

    residual = Conv1D(1024, 1, strides=2, padding='same')(previous_block_activation)
    x = Add()([x, residual])

    x = Activation('relu')(x)
    x = SeparableConv1D(728, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv1D(1024, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(1, activation='linear')(x)

    return x

def xception(input_shape):
    inputs = Input(shape=input_shape)
    outputs = xception_exit_flow(xception_middle_flow(xception_entry_flow(inputs)))
    return Model(inputs, outputs)

def nicon_vg_big(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(SpatialDropout1D(0.2))
    model.add(Conv1D (filters=64, kernel_size=3, padding="same", activation='swish'))
    model.add(Conv1D (filters=64, kernel_size=3, padding="same", activation='swish'))
    model.add(MaxPool1D(pool_size=5,strides=3))
    model.add(SpatialDropout1D(0.2))
    model.add(Conv1D (filters=128, kernel_size=3, padding="same", activation='swish'))
    model.add(Conv1D (filters=128, kernel_size=3, padding="same", activation='swish'))
    model.add(MaxPool1D(pool_size=5,strides=3))
    model.add(SpatialDropout1D(0.2))
    model.add(Flatten())
    model.add(Dense(units=2096, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=2096, activation="relu"))
    model.add(Dense(units=1, activation="sigmoid"))
    # we compile the model with the custom Adam optimizer
    model.compile(loss = 'mean_squared_error', metrics=['mse'], optimizer = "adam")
    # model.summary()
    return model
    

def nicon_vg(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(SpatialDropout1D(0.2))
    model.add(Conv1D (filters=64, kernel_size=3, padding="same", activation='swish'))
    model.add(Conv1D (filters=64, kernel_size=3, padding="same", activation='swish'))
    model.add(MaxPool1D(pool_size=5,strides=3))
    model.add(SpatialDropout1D(0.2))
    model.add(Conv1D (filters=128, kernel_size=3, padding="same", activation='swish'))
    model.add(Conv1D (filters=128, kernel_size=3, padding="same", activation='swish'))
    model.add(MaxPool1D(pool_size=5,strides=3))
    model.add(SpatialDropout1D(0.2))
    model.add(Flatten())
    model.add(Dense(units=2096, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=2096, activation="relu"))
    model.add(Dense(units=1, activation="sigmoid"))
    # we compile the model with the custom Adam optimizer
    model.compile(loss = 'mean_squared_error', metrics=['mse'], optimizer = "adam")
    return model
    
    
def nicon(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(SpatialDropout1D(0.08))
    model.add(Conv1D (filters=8, kernel_size=15, strides=5, activation='selu'))
    model.add(Dropout(0.2))
    model.add(Conv1D (filters=64, kernel_size=21, strides=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv1D (filters=32, kernel_size=5, strides=3, activation='elu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    return model


# def chaix(meta: Dict[str, Any]):
#     input_shape = meta["X_shape_"][1:]
#     model = Sequential()
#     model.add(Input(shape=input_shape))
#     model.add(SpatialDropout1D(0.2))
#     model.add(DepthwiseConv1D(kernel_size=7, padding="same",depth_multiplier=2, activation='relu'))
#     model.add(DepthwiseConv1D(kernel_size=7, padding="same",depth_multiplier=2, activation='relu'))
#     model.add(MaxPooling1D(pool_size=2,strides=2))
#     model.add(BatchNormalization())
#     model.add(DepthwiseConv1D(kernel_size=5, padding="same",depth_multiplier=2, activation='relu'))
#     model.add(DepthwiseConv1D(kernel_size=5, padding="same",depth_multiplier=2, activation='relu'))
#     model.add(MaxPooling1D(pool_size=2,strides=2))
#     model.add(BatchNormalization())
#     model.add(DepthwiseConv1D(kernel_size=9, padding="same",depth_multiplier=2, activation='relu'))
#     model.add(DepthwiseConv1D(kernel_size=9, padding="same",depth_multiplier=2, activation='relu'))
#     model.add(MaxPooling1D(pool_size=2,strides=2))
#     model.add(BatchNormalization())
    
#     model.add(SeparableConv1D(64, kernel_size=3, depth_multiplier=1, padding="same", activation='relu'))
#     model.add(Conv1D(filters=32, kernel_size=3, padding="same"))
#     model.add(MaxPooling1D(pool_size=5,strides=3))
#     model.add(SpatialDropout1D(0.1))
#     model.add(Flatten())
#     model.add(Dense(units=128, activation="relu"))
#     model.add(Dense(units=32, activation="relu"))
#     model.add(Dropout(0.2))
#     model.add(Dense(units=1, activation="sigmoid"))
#     model.compile(loss = 'mean_squared_error', metrics=['mse'], optimizer = "rmsprop")
#     model.summary()
#     return model