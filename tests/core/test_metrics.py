# import pytest
# from pinard.core.metrics import get_metric
# from sklearn.metrics import mean_squared_error
# from tensorflow.keras.metrics import MeanSquaredError


# def test_get_metric_sklearn():
#     metric = get_metric('neg_mean_squared_error', framework='sklearn')
#     assert metric == mean_squared_error


# def test_get_metric_tensorflow():
#     metric = get_metric('neg_mean_squared_error', framework='tensorflow')
#     assert isinstance(metric, type(MeanSquaredError))


# def test_get_metric_unknown():
#     with pytest.raises(ValueError):
#         get_metric('unknown_metric', framework='sklearn')
