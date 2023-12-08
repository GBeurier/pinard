import numpy as np
import pytest

from pinard import preprocessing as pp

preprocessings = [
    (0, "IdentityTransformer", pp.IdentityTransformer()),
    (1, "Baseline", pp.Baseline()),
    (2, "StandardNormalVariate", pp.StandardNormalVariate()),
    (3, "RobustNormalVariate", pp.RobustNormalVariate()),
    (4, "SavitzkyGolay", pp.SavitzkyGolay()),
    (5, "Normalize", pp.Normalize()),
    (6, "Detrend", pp.Detrend()),
    (7, "MultiplicativeScatterCorrection", pp.MultiplicativeScatterCorrection()),
    (8, "Derivate", pp.Derivate()),
    (9, "Gaussian", pp.Gaussian()),
    (10, "Wavelet", pp.Wavelet()),
    (11, "Haar", pp.Haar()),
    (12, "SimpleScale", pp.SimpleScale()),
]


@pytest.mark.parametrize("index, name, preprocess", preprocessings)
def test_preprocessings(
    sample_data,
    preprocessing_validation_data,
    index,
    name,
    preprocess,
):
    preprocess.fit(sample_data)
    transformed_data = preprocess.transform(sample_data)
    x = transformed_data[0]
    np.testing.assert_array_almost_equal(
        x, preprocessing_validation_data[index], decimal=5
    )

    if hasattr(preprocess, "inverse_transform"):
        reverse_data = preprocess.inverse_transform(transformed_data)
        np.testing.assert_array_almost_equal(sample_data, reverse_data, decimal=5)

    transformed_data = preprocess.fit_transform(sample_data)
    x = transformed_data[0]
    np.testing.assert_array_almost_equal(
        x, preprocessing_validation_data[index], decimal=5
    )


preprocessing_func = [
    ("baseline", pp.baseline, 1),
    ("savgol", pp.savgol, 4),
    ("norml", pp.norml, 5),
    ("baseline", pp.detrend, 6),
    ("msc", pp.msc, 7),
    ("derivate", pp.derivate, 8),
    ("baseline", pp.gaussian, 9),
    ("spl_norml", pp.spl_norml, 12),
]


@pytest.mark.parametrize("name, preprocess, index", preprocessing_func)
def test_preprocessings_functions(
    sample_data,
    preprocessing_validation_data,
    name,
    preprocess,
    index,
):
    transformed_data = preprocess(sample_data)
    x = transformed_data[0]
    np.testing.assert_array_almost_equal(
        x, preprocessing_validation_data[index], decimal=5
    )
