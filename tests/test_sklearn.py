import numpy as np

import pinard.augmentation as aug
from pinard.sklearn import FeatureAugmentation, SampleAugmentation


def test_features_augmentation(
    sample_data, preprocessing_validation_data, default_preprocessings
):
    pipeline = FeatureAugmentation(default_preprocessings)
    x_proc = pipeline.fit_transform(sample_data)
    for i in range(len(default_preprocessings)):
        np.testing.assert_array_almost_equal(
            x_proc[0, :, i], preprocessing_validation_data[i], decimal=5
        )


augmenters = [
    (1, "id", aug.IdentityAugmenter()),
    (6, "rotate_translate", aug.Rotate_Translate()),
    (3, "Rotate_Translate_custom", aug.Rotate_Translate(p_range=5, y_factor=5)),
    (2, "Random_X_Operation", aug.Random_X_Operation()),
]


def test_samples_augmentation(
    sample_data, preprocessing_validation_data, default_preprocessings
):
    x = sample_data[:, 1:]
    y = sample_data[:, 0]
    augmentation_count = sum([aug[0] for aug in augmenters])
    pipeline = SampleAugmentation(augmenters)
    x_aug, y_aug = pipeline.transform(x, y)
    assert x_aug.shape[0] == augmentation_count * x.shape[0]
    assert y_aug.shape[0] == augmentation_count * y.shape[0]
    assert x_aug.shape[1] == x.shape[1]
    for i in range(x.shape[0]):
        np.testing.assert_array_almost_equal(
            x[i], x_aug[i * augmentation_count], decimal=5
        )
