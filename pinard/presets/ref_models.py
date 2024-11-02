import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    DepthwiseConv1D,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
    Input,
    LayerNormalization,
    MaxPooling1D,
    MultiHeadAttention,
    SeparableConv1D,
    SpatialDropout1D,
)

from keras.models import Model, Sequential
from pinard.core.utils import framework


@framework('tensorflow')
def decon(input_shape, params={}):
    """
    Builds a CNN model with depthwise and separable convolutions.

    Parameters:
        input_shape (tuple): Shape of the input data.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Sequential: Compiled model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(SpatialDropout1D(params.get('spatial_dropout', 0.2)))
    model.add(DepthwiseConv1D(
        kernel_size=7,
        padding="same",
        depth_multiplier=2,
        activation="relu"
    ))
    model.add(DepthwiseConv1D(
        kernel_size=7,
        padding="same",
        depth_multiplier=2,
        activation="relu"
    ))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(BatchNormalization())
    model.add(DepthwiseConv1D(
        kernel_size=5,
        padding="same",
        depth_multiplier=2,
        activation="relu"
    ))
    model.add(DepthwiseConv1D(
        kernel_size=5,
        padding="same",
        depth_multiplier=2,
        activation="relu"
    ))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(BatchNormalization())
    model.add(DepthwiseConv1D(
        kernel_size=9,
        padding="same",
        depth_multiplier=2,
        activation="relu"
    ))
    model.add(DepthwiseConv1D(
        kernel_size=9,
        padding="same",
        depth_multiplier=2,
        activation="relu"
    ))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(BatchNormalization())
    model.add(SeparableConv1D(
        64,
        kernel_size=3,
        depth_multiplier=1,
        padding="same",
        activation="relu"
    ))
    model.add(Conv1D(
        filters=32,
        kernel_size=3,
        padding="same",
        activation="relu"
    ))
    model.add(MaxPooling1D(pool_size=5, strides=3))
    model.add(SpatialDropout1D(0.1))
    model.add(Flatten())
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=32, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation="sigmoid"))
    return model


def decon_Sep(input_shape, params={}):
    """
    Builds a CNN model with separable convolutions.

    Parameters:
        input_shape (tuple): Shape of the input data.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Sequential: Compiled model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(SpatialDropout1D(params.get('spatial_dropout', 0.2)))
    model.add(SeparableConv1D(
        filters=params.get('filters1', 64),
        kernel_size=params.get('kernel_size1', 3),
        strides=params.get('strides1', 2),
        depth_multiplier=params.get('depth_multiplier1', 32),
        padding="same",
        activation="relu"
    ))
    model.add(BatchNormalization())
    model.add(SeparableConv1D(
        filters=params.get('filters2', 64),
        kernel_size=params.get('kernel_size2', 3),
        strides=params.get('strides2', 2),
        depth_multiplier=params.get('depth_multiplier2', 32),
        padding="same",
        activation="relu"
    ))
    model.add(BatchNormalization())
    model.add(SeparableConv1D(
        filters=params.get('filters3', 64),
        kernel_size=params.get('kernel_size3', 3),
        depth_multiplier=params.get('depth_multiplier3', 32),
        padding="same",
        activation="relu"
    ))
    model.add(BatchNormalization())
    model.add(SeparableConv1D(
        filters=params.get('filters4', 64),
        kernel_size=params.get('kernel_size4', 3),
        depth_multiplier=params.get('depth_multiplier4', 32),
        padding="same",
        activation="relu"
    ))
    model.add(BatchNormalization())
    model.add(Conv1D(
        filters=params.get('filters5', 32),
        kernel_size=params.get('kernel_size5', 5),
        strides=params.get('strides5', 2),
        padding="same",
        activation="relu"
    ))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(units=params.get('dense_units', 32), activation="relu"))
    model.add(Dropout(params.get('dropout_rate', 0.2)))
    model.add(Dense(units=1, activation="sigmoid"))
    return model


@framework('tensorflow')
def nicon(input_shape, params={}):
    """
    Builds a custom CNN model with depthwise convolutions.

    Parameters:
        input_shape (tuple): Shape of the input data.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Sequential: Compiled model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(SpatialDropout1D(params.get('spatial_dropout', 0.08)))
    model.add(Conv1D(filters=params.get('filters1', 8), kernel_size=15, strides=5, activation="selu"))
    model.add(Dropout(params.get('dropout_rate', 0.2)))
    model.add(Conv1D(filters=params.get('filters2', 64), kernel_size=21, strides=3, activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=params.get('filters3', 32), kernel_size=5, strides=3, activation="elu"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(params.get('dense_units', 16), activation="sigmoid"))
    model.add(Dense(1, activation="sigmoid"))
    return model


@framework('tensorflow')
def customizable_nicon(input_shape, params={}):
    """
    Builds a custom CNN model with depthwise convolutions.

    Parameters:
        input_shape (tuple): Shape of the input data.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Sequential: Compiled model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(SpatialDropout1D(params.get('spatial_dropout', 0.08)))
    model.add(Conv1D(filters=params.get('filters1', 8), kernel_size=params.get('kernel_size1', 15), strides=params.get('strides1', 5), activation=params.get('activation1', "selu")))
    model.add(Dropout(params.get('dropout_rate', 0.2)))
    model.add(Conv1D(filters=params.get('filters2', 64), kernel_size=params.get('kernel_size2', 21), strides=params.get('strides2', 3), activation=params.get('activation2', "relu")))
    model.add(BatchNormalization() if params.get('normalization_method1', "BatchNormalization") == "BatchNormalization" else LayerNormalization())
    model.add(Conv1D(filters=params.get('filters3', 32), kernel_size=params.get('kernel_size3', 5), strides=params.get('strides3', 3), activation=params.get('activation3', "elu")))
    model.add(BatchNormalization() if params.get('normalization_method2', "BatchNormalization") == "BatchNormalization" else LayerNormalization())
    model.add(Flatten())
    model.add(Dense(params.get('dense_units', 16), activation=params.get('dense_activation', "sigmoid")))
    model.add(Dense(1, activation="sigmoid"))
    return model

nicon_sample_finetune = {
    'spatial_dropout': (float, 0.01, 0.5),
    'filters1': [4, 8, 16, 32, 64, 128, 256],
    'kernel_size1': [3, 5, 7, 9, 11, 13, 15],
    'strides1': [1, 2, 3, 4, 5],
    'activation1': ['relu', 'selu', 'elu', 'swish'],
    'dropout_rate': (float, 0.01, 0.5),
    'filters2': [4, 8, 16, 32, 64, 128, 256],
    'kernel_size2': [3, 5, 7, 9, 11, 13, 15],
    'strides2': [1, 2, 3, 4, 5],
    'activation2': ['relu', 'selu', 'elu', 'swish'],
    'normalization_method1': ['BatchNormalization', 'LayerNormalization'],
    'filters3': [4, 8, 16, 32, 64, 128, 256],
    'kernel_size3': [3, 5, 7, 9, 11, 13, 15],
    'strides3': [1, 2, 3, 4, 5],
    'activation3': ['relu', 'selu', 'elu', 'swish'],
    'normalization_method2': ['BatchNormalization', 'LayerNormalization'],
    'dense_units': [4, 8, 16, 32, 64, 128, 256],
    'dense_activation': ['relu', 'selu', 'elu', 'swish'],
}


@framework('tensorflow')
def nicon_VG(input_shape, params={}):
    """
    Builds a custom CNN model.

    Parameters:
        input_shape (tuple): Shape of the input data.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Sequential: Compiled CNN model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(SpatialDropout1D(params.get('spatial_dropout', 0.2)))
    model.add(Conv1D(filters=params.get('filters1', 64), kernel_size=3, padding="same", activation="swish"))
    model.add(Conv1D(filters=params.get('filters2', 64), kernel_size=3, padding="same", activation="swish"))
    model.add(MaxPooling1D(pool_size=5, strides=3))
    model.add(SpatialDropout1D(params.get('spatial_dropout', 0.2)))
    model.add(Conv1D(filters=params.get('filters3', 128), kernel_size=3, padding="same", activation="swish"))
    model.add(Conv1D(filters=params.get('filters4', 128), kernel_size=3, padding="same", activation="swish"))
    model.add(MaxPooling1D(pool_size=5, strides=3))
    model.add(SpatialDropout1D(params.get('spatial_dropout', 0.2)))
    model.add(Flatten())
    model.add(Dense(units=params.get('dense_units1', 1024), activation="relu"))
    model.add(Dropout(params.get('dropout_rate', 0.2)))
    model.add(Dense(units=params.get('dense_units2', 1024), activation="relu"))
    model.add(Dense(units=1, activation="sigmoid"))
    return model


@framework('tensorflow')
def decon_layer(input_shape, params={}):
    """
    Builds a model using depthwise separable convolutions and layer normalization.

    Parameters:
        input_shape (tuple): Shape of the input data.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Model: Compiled deconvolutional model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(SpatialDropout1D(params.get('spatial_dropout', 0.2)))
    model.add(DepthwiseConv1D(
        kernel_size=params.get('kernel_size1', 7), padding="same", depth_multiplier=2, activation="relu"
    ))
    model.add(DepthwiseConv1D(
        kernel_size=params.get('kernel_size2', 7), padding="same", depth_multiplier=2, activation="relu"
    ))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(LayerNormalization())

    model.add(DepthwiseConv1D(
        kernel_size=params.get('kernel_size3', 5), padding="same", depth_multiplier=2, activation="relu"
    ))
    model.add(DepthwiseConv1D(
        kernel_size=params.get('kernel_size4', 5), padding="same", depth_multiplier=2, activation="relu"
    ))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(LayerNormalization())

    model.add(DepthwiseConv1D(
        kernel_size=params.get('kernel_size5', 9), padding="same", depth_multiplier=2, activation="relu"
    ))
    model.add(DepthwiseConv1D(
        kernel_size=params.get('kernel_size6', 9), padding="same", depth_multiplier=2, activation="relu"
    ))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(LayerNormalization())

    model.add(SeparableConv1D(64, kernel_size=3, depth_multiplier=1, padding="same", activation="relu"))
    model.add(Conv1D(filters=32, kernel_size=3, padding="same"))
    model.add(MaxPooling1D(pool_size=5, strides=3))
    model.add(SpatialDropout1D(params.get('spatial_dropout', 0.1)))
    model.add(Flatten())

    # Fully Connected layers
    model.add(Dense(units=params.get('dense_units1', 128), activation="relu"))
    model.add(Dense(units=params.get('dense_units2', 32), activation="relu"))
    model.add(Dropout(params.get('dropout_rate', 0.2)))

    # Output layer
    model.add(Dense(units=1, activation="sigmoid"))

    return model


def transformer_model(input_shape, params={}):
    """
    Builds a transformer model for 1D data.

    Parameters:
        input_shape (tuple): Shape of the input data.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Model: Compiled transformer model.
    """
    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
        x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward block
        x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = Dropout(dropout)(x)
        x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        return x + res

    inputs = Input(shape=input_shape)
    x = inputs

    # Stacking Transformer blocks
    for _ in range(params.get('num_transformer_blocks', 1)):
        x = transformer_encoder(
            x,
            head_size=params.get('head_size', 16),
            num_heads=params.get('num_heads', 2),
            ff_dim=params.get('ff_dim', 8),
            dropout=params.get('dropout', 0.05),
        )

    x = GlobalAveragePooling1D(data_format="channels_first")(x)

    # Fully connected layers after transformer blocks
    for dim in params.get('mlp_units', [32, 8]):
        x = Dense(dim, activation="relu")(x)
        x = Dropout(params.get('mlp_dropout', 0.1))(x)

    outputs = Dense(units=1, activation="sigmoid")(x)
    return Model(inputs, outputs)


@framework('tensorflow')
def transformer_VG(input_shape, params={}):
    return transformer_model(input_shape, {
                            'head_size': params.get('head_size', 16),
                            'num_heads': params.get('num_heads', 32),
                            'ff_dim': params.get('ff_dim', 8),
                            'num_transformer_blocks': params.get('num_transformer_blocks', 1),
                            'mlp_units': params.get('mlp_units', [32, 8]),
                            'dropout': params.get('dropout', 0.05),
                            'mlp_dropout': params.get('mlp_dropout', 0.1),
                        })


@framework('tensorflow')
def transformer(input_shape, params={}):
    return transformer_model(input_shape, {
                            'head_size': params.get('head_size', 8),
                            'num_heads': params.get('num_heads', 2),
                            'ff_dim': params.get('ff_dim', 4),
                            'num_transformer_blocks': params.get('num_transformer_blocks', 1),
                            'mlp_units': params.get('mlp_units', [8]),
                            'dropout': params.get('dropout', 0.05),
                            'mlp_dropout': params.get('mlp_dropout', 0.1),
                        })
    
    
@framework('tensorflow')
def decon_classification(input_shape, num_classes=2, params={}):
    """
    Builds a CNN model with depthwise and separable convolutions for classification.

    Parameters:
        input_shape (tuple): Shape of the input data.
        num_classes (int): Number of classes for classification.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Sequential: Compiled classification model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(SpatialDropout1D(params.get('spatial_dropout', 0.2)))
    model.add(DepthwiseConv1D(
        kernel_size=7,
        padding="same",
        depth_multiplier=2,
        activation="relu"
    ))
    model.add(DepthwiseConv1D(
        kernel_size=7,
        padding="same",
        depth_multiplier=2,
        activation="relu"
    ))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(BatchNormalization())
    model.add(DepthwiseConv1D(
        kernel_size=5,
        padding="same",
        depth_multiplier=2,
        activation="relu"
    ))
    model.add(DepthwiseConv1D(
        kernel_size=5,
        padding="same",
        depth_multiplier=2,
        activation="relu"
    ))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(BatchNormalization())
    model.add(DepthwiseConv1D(
        kernel_size=9,
        padding="same",
        depth_multiplier=2,
        activation="relu"
    ))
    model.add(DepthwiseConv1D(
        kernel_size=9,
        padding="same",
        depth_multiplier=2,
        activation="relu"
    ))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(BatchNormalization())
    model.add(SeparableConv1D(
        64,
        kernel_size=3,
        depth_multiplier=1,
        padding="same",
        activation="relu"
    ))
    model.add(Conv1D(
        filters=32,
        kernel_size=3,
        padding="same",
        activation="relu"
    ))
    model.add(MaxPooling1D(pool_size=5, strides=3))
    model.add(SpatialDropout1D(0.1))
    model.add(Flatten())
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=32, activation="relu"))
    model.add(Dropout(0.2))
    if num_classes == 2:
        model.add(Dense(units=1, activation="sigmoid"))
    else:
        model.add(Dense(units=num_classes, activation="softmax"))
    return model


def decon_Sep_classification(input_shape, num_classes=2, params={}):
    """
    Builds a CNN model with separable convolutions for classification.

    Parameters:
        input_shape (tuple): Shape of the input data.
        num_classes (int): Number of classes for classification.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Sequential: Compiled classification model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(SpatialDropout1D(params.get('spatial_dropout', 0.2)))
    model.add(SeparableConv1D(
        filters=params.get('filters1', 64),
        kernel_size=params.get('kernel_size1', 3),
        strides=params.get('strides1', 2),
        depth_multiplier=params.get('depth_multiplier1', 32),
        padding="same",
        activation="relu"
    ))
    model.add(BatchNormalization())
    model.add(SeparableConv1D(
        filters=params.get('filters2', 64),
        kernel_size=params.get('kernel_size2', 3),
        strides=params.get('strides2', 2),
        depth_multiplier=params.get('depth_multiplier2', 32),
        padding="same",
        activation="relu"
    ))
    model.add(BatchNormalization())
    model.add(SeparableConv1D(
        filters=params.get('filters3', 64),
        kernel_size=params.get('kernel_size3', 3),
        depth_multiplier=params.get('depth_multiplier3', 32),
        padding="same",
        activation="relu"
    ))
    model.add(BatchNormalization())
    model.add(SeparableConv1D(
        filters=params.get('filters4', 64),
        kernel_size=params.get('kernel_size4', 3),
        depth_multiplier=params.get('depth_multiplier4', 32),
        padding="same",
        activation="relu"
    ))
    model.add(BatchNormalization())
    model.add(Conv1D(
        filters=params.get('filters5', 32),
        kernel_size=params.get('kernel_size5', 5),
        strides=params.get('strides5', 2),
        padding="same",
        activation="relu"
    ))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(units=params.get('dense_units', 32), activation="relu"))
    model.add(Dropout(params.get('dropout_rate', 0.2)))
    if num_classes >= 2:
        model.add(Dense(units=1, activation="sigmoid"))
    else:
        model.add(Dense(units=num_classes, activation="softmax"))
    return model


@framework('tensorflow')
def nicon_classification(input_shape, num_classes=2, params={}):
    """
    Builds a custom CNN model with depthwise convolutions for classification.

    Parameters:
        input_shape (tuple): Shape of the input data.
        num_classes (int): Number of classes for classification.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Sequential: Compiled classification model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(SpatialDropout1D(params.get('spatial_dropout', 0.08)))
    model.add(Conv1D(filters=params.get('filters1', 8), kernel_size=15, strides=5, activation="selu"))
    model.add(Dropout(params.get('dropout_rate', 0.2)))
    model.add(Conv1D(filters=params.get('filters2', 64), kernel_size=21, strides=3, activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=params.get('filters3', 32), kernel_size=5, strides=3, activation="elu"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(params.get('dense_units', 16), activation="sigmoid"))
    if num_classes == 2:
        model.add(Dense(1, activation="sigmoid"))
    else:
        model.add(Dense(units=num_classes, activation="softmax"))
    return model


@framework('tensorflow')
def customizable_nicon_classification(input_shape, num_classes=2, params={}):
    """
    Builds a custom CNN model with depthwise convolutions for classification.

    Parameters:
        input_shape (tuple): Shape of the input data.
        num_classes (int): Number of classes for classification.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Sequential: Compiled classification model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(SpatialDropout1D(params.get('spatial_dropout', 0.08)))
    model.add(Conv1D(filters=params.get('filters1', 8), kernel_size=params.get('kernel_size1', 15), strides=params.get('strides1', 5), activation=params.get('activation1', "selu")))
    model.add(Dropout(params.get('dropout_rate', 0.2)))
    model.add(Conv1D(filters=params.get('filters2', 64), kernel_size=params.get('kernel_size2', 21), strides=params.get('strides2', 3), activation=params.get('activation2', "relu")))
    model.add(BatchNormalization() if params.get('normalization_method1', "BatchNormalization") == "BatchNormalization" else LayerNormalization())
    model.add(Conv1D(filters=params.get('filters3', 32), kernel_size=params.get('kernel_size3', 5), strides=params.get('strides3', 3), activation=params.get('activation3', "elu")))
    model.add(BatchNormalization() if params.get('normalization_method2', "BatchNormalization") == "BatchNormalization" else LayerNormalization())
    model.add(Flatten())
    model.add(Dense(params.get('dense_units', 16), activation=params.get('dense_activation', "sigmoid")))
    if num_classes == 2:
        model.add(Dense(1, activation="sigmoid"))
    else:
        model.add(Dense(units=num_classes, activation="softmax"))
    return model


@framework('tensorflow')
def nicon_VG_classification(input_shape, num_classes=2, params={}):
    """
    Builds a custom CNN model for classification.

    Parameters:
        input_shape (tuple): Shape of the input data.
        num_classes (int): Number of classes for classification.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Sequential: Compiled classification model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(SpatialDropout1D(params.get('spatial_dropout', 0.2)))
    model.add(Conv1D(filters=params.get('filters1', 64), kernel_size=3, padding="same", activation="swish"))
    model.add(Conv1D(filters=params.get('filters2', 64), kernel_size=3, padding="same", activation="swish"))
    model.add(MaxPooling1D(pool_size=5, strides=3))
    model.add(SpatialDropout1D(params.get('spatial_dropout', 0.2)))
    model.add(Conv1D(filters=params.get('filters3', 128), kernel_size=3, padding="same", activation="swish"))
    model.add(Conv1D(filters=params.get('filters4', 128), kernel_size=3, padding="same", activation="swish"))
    model.add(MaxPooling1D(pool_size=5, strides=3))
    model.add(SpatialDropout1D(params.get('spatial_dropout', 0.2)))
    model.add(Flatten())
    model.add(Dense(units=params.get('dense_units1', 1024), activation="relu"))
    model.add(Dropout(params.get('dropout_rate', 0.2)))
    model.add(Dense(units=params.get('dense_units2', 1024), activation="relu"))
    if num_classes == 2:
        model.add(Dense(units=1, activation="sigmoid"))
    else:
        model.add(Dense(units=num_classes, activation="softmax"))
    return model


@framework('tensorflow')
def decon_layer_classification(input_shape, num_classes=2, params={}):
    """
    Builds a model using depthwise separable convolutions and layer normalization for classification.

    Parameters:
        input_shape (tuple): Shape of the input data.
        num_classes (int): Number of classes for classification.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Model: Compiled classification model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(SpatialDropout1D(params.get('spatial_dropout', 0.2)))
    model.add(DepthwiseConv1D(
        kernel_size=params.get('kernel_size1', 7), padding="same", depth_multiplier=2, activation="relu"
    ))
    model.add(DepthwiseConv1D(
        kernel_size=params.get('kernel_size2', 7), padding="same", depth_multiplier=2, activation="relu"
    ))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(LayerNormalization())

    model.add(DepthwiseConv1D(
        kernel_size=params.get('kernel_size3', 5), padding="same", depth_multiplier=2, activation="relu"
    ))
    model.add(DepthwiseConv1D(
        kernel_size=params.get('kernel_size4', 5), padding="same", depth_multiplier=2, activation="relu"
    ))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(LayerNormalization())

    model.add(DepthwiseConv1D(
        kernel_size=params.get('kernel_size5', 9), padding="same", depth_multiplier=2, activation="relu"
    ))
    model.add(DepthwiseConv1D(
        kernel_size=params.get('kernel_size6', 9), padding="same", depth_multiplier=2, activation="relu"
    ))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(LayerNormalization())

    model.add(SeparableConv1D(64, kernel_size=3, depth_multiplier=1, padding="same", activation="relu"))
    model.add(Conv1D(filters=32, kernel_size=3, padding="same"))
    model.add(MaxPooling1D(pool_size=5, strides=3))
    model.add(SpatialDropout1D(params.get('spatial_dropout', 0.1)))
    model.add(Flatten())

    # Fully Connected layers
    model.add(Dense(units=params.get('dense_units1', 128), activation="relu"))
    model.add(Dense(units=params.get('dense_units2', 32), activation="relu"))
    model.add(Dropout(params.get('dropout_rate', 0.2)))

    # Output layer
    if num_classes == 2:
        model.add(Dense(units=1, activation="sigmoid"))
    else:
        model.add(Dense(units=num_classes, activation="softmax"))

    return model


def transformer_model_classification(input_shape, num_classes=2, params={}):
    """
    Builds a transformer model for 1D data classification.

    Parameters:
        input_shape (tuple): Shape of the input data.
        num_classes (int): Number of classes for classification.
        params (dict): Dictionary of parameters for model configuration.

    Returns:
        keras.Model: Compiled transformer classification model.
    """
    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
        x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward block
        x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = Dropout(dropout)(x)
        x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        return x + res

    inputs = Input(shape=input_shape)
    x = inputs

    # Stacking Transformer blocks
    for _ in range(params.get('num_transformer_blocks', 1)):
        x = transformer_encoder(
            x,
            head_size=params.get('head_size', 16),
            num_heads=params.get('num_heads', 2),
            ff_dim=params.get('ff_dim', 8),
            dropout=params.get('dropout', 0.05),
        )

    x = GlobalAveragePooling1D(data_format="channels_first")(x)

    # Fully connected layers after transformer blocks
    for dim in params.get('mlp_units', [32, 8]):
        x = Dense(dim, activation="relu")(x)
        x = Dropout(params.get('mlp_dropout', 0.1))(x)

    if num_classes == 2:
        outputs = Dense(units=1, activation="sigmoid")(x)
    else:
        outputs = Dense(units=num_classes, activation="softmax")(x)

    return Model(inputs, outputs)


@framework('tensorflow')
def transformer_VG_classification(input_shape, num_classes=2, params={}):
    return transformer_model_classification(input_shape, num_classes, {
        'head_size': params.get('head_size', 16),
        'num_heads': params.get('num_heads', 32),
        'ff_dim': params.get('ff_dim', 8),
        'num_transformer_blocks': params.get('num_transformer_blocks', 1),
        'mlp_units': params.get('mlp_units', [32, 8]),
        'dropout': params.get('dropout', 0.05),
        'mlp_dropout': params.get('mlp_dropout', 0.1),
    })


@framework('tensorflow')
def transformer_classification(input_shape, num_classes=2, params={}):
    return transformer_model_classification(input_shape, num_classes, {
        'head_size': params.get('head_size', 8),
        'num_heads': params.get('num_heads', 2),
        'ff_dim': params.get('ff_dim', 4),
        'num_transformer_blocks': params.get('num_transformer_blocks', 1),
        'mlp_units': params.get('mlp_units', [8]),
        'dropout': params.get('dropout', 0.05),
        'mlp_dropout': params.get('mlp_dropout', 0.1),
    })


# def build_model(input_shape, params, task_type='regression', num_classes=1):
#     # ... build your model layers ...

#     # Adjust the output layer based on the task type
#     if task_type == 'classification':
#         if num_classes == 2:
#             activation = 'sigmoid'
#             units = 1
#         else:
#             activation = 'softmax'
#             units = num_classes
#     else:
#         activation = 'linear'
#         units = 1

#     model.add(Dense(units=units, activation=activation))
#     return model
