import numpy as np
import pytest
import operator

from pinard import augmentation as aug

seed = 42
augmenters = [
    (0, "Rotate_Translate", aug.Rotate_Translate, {"random_state": seed}),
    (1, "Rotate_Translate_custom", aug.Rotate_Translate, {"random_state": seed, "p_range": 5, "y_factor": 5}),
    (2, "Random_X_Operation", aug.Random_X_Operation, {"random_state": seed}),
    (3, "Random_X_Operation_custom", aug.Random_X_Operation, {"random_state": seed, "operator_func": operator.add, "operator_range": (-0.002, 0.002)}),
    (4, "Spline_Smoothing", aug.Spline_Smoothing, {"random_state": seed}),
    (5, "Spline_X_Perturbation", aug.Spline_X_Perturbations, {"random_state": seed}),
    (6, "Spline_X_Perturbation", aug.Spline_X_Perturbations, {"random_state": seed, "perturbation_density": 0.01, "perturbation_range": (-30, 30)}),
    (7, "Spline_Y_Perturbation", aug.Spline_Y_Perturbations, {"random_state": seed}),
    (8, "Spline_Y_Perturbation", aug.Spline_Y_Perturbations, {"random_state": seed, "spline_points": 5, "perturbation_intensity": 0.02}),
    (9, "Spline_X_Simplification", aug.Spline_X_Simplification, {"random_state": seed}),
    (10, "Spline_X_Simplification", aug.Spline_X_Simplification, {"random_state": seed, "spline_points": 15}),
    (11, "Spline_Curve_Simplification", aug.Spline_Curve_Simplification, {"random_state": seed}),
    (12, "Spline_Curve_Simplification", aug.Spline_Curve_Simplification, {"random_state": seed, "spline_points": 5, "uniform": False}),
    (13, "Spline_Curve_Simplification", aug.Spline_Curve_Simplification, {"random_state": seed, "spline_points": 5, "uniform": True}),
]


@pytest.mark.parametrize("index, name, augmenter, params", augmenters)
def test_global_augmented_data(augmentation_source_data, index, name, augmenter, params, augmented_global_data_csv, augmented_samples_data_csv):
    aug_instance = augmenter(**params)
    g_augmented_data = aug_instance.augment(augmentation_source_data, apply_on="global")
    np.testing.assert_array_almost_equal(g_augmented_data, augmented_global_data_csv, decimal=3)


@pytest.mark.parametrize("index, name, augmenter, params", augmenters)
def test_samples_augmented_data(augmentation_source_data, index, name, augmenter, params, augmented_global_data_csv, augmented_samples_data_csv):
    aug_instance = augmenter(**params)
    s_augmented_data = aug_instance.augment(augmentation_source_data, apply_on="samples")
    np.testing.assert_array_almost_equal(s_augmented_data, augmented_samples_data_csv, decimal=3)
