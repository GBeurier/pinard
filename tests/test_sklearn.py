import numpy as np

from pinard.sklearn import FeatureAugmentation


def test_preprocessings(
    sample_data, preprocessing_validation_data, default_preprocessings
):
    pipeline = FeatureAugmentation(default_preprocessings)
    x_proc = pipeline.fit_transform(sample_data)
    print(x_proc.shape)
    for i in range(len(default_preprocessings)):
        np.testing.assert_array_almost_equal(
            x_proc[0, :, i], preprocessing_validation_data[i], decimal=5
        )
