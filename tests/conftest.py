from os.path import abspath
from os.path import dirname as d
import operator

import numpy as np
import pytest

from pinard import preprocessing as pp
from pinard import augmentation as aug

ROOT_DIR = d(d(abspath(__file__)))


def path_to(str):
    return ROOT_DIR + "/tests/data/" + str


@pytest.fixture(scope="module")
def simple_data():
    return path_to("test_load_src.csv")


@pytest.fixture(scope="module")
def simple_data_na():
    return path_to("test_load_na.csv")


@pytest.fixture(scope="module")
def bad_data():
    return path_to("test_bad_data.csv")


@pytest.fixture(scope="module")
def split_data():
    split_data = np.loadtxt(path_to("test_split.csv"), delimiter=";")
    y = np.reshape(split_data[:, 0], (-1, 1))
    x = split_data[:, 1:]
    return (x, y)


@pytest.fixture(scope="module")
def split_validation_data():
    split_validation_data = np.loadtxt(
        path_to("test_split_validation.csv"), delimiter=";"
    )
    return split_validation_data.astype(int)


@pytest.fixture(scope="session")
def sample_data():
    sample_data = np.loadtxt(path_to("test_data.csv"), delimiter=";")
    return sample_data


@pytest.fixture(scope="session")
def preprocessing_validation_data():
    validation_data = np.loadtxt(
        path_to("test_preprocessing_validation.csv"), delimiter=";"
    )
    return validation_data


@pytest.fixture(scope="session")
def default_preprocessings():
    preprocessings = [
        ("IdentityTransformer", pp.IdentityTransformer()),
        ("Baseline", pp.Baseline()),
        ("StandardNormalVariate", pp.StandardNormalVariate()),
        ("RobustNormalVariate", pp.RobustNormalVariate()),
        ("SavitzkyGolay", pp.SavitzkyGolay()),
        ("Normalize", pp.Normalize()),
        ("Detrend", pp.Detrend()),
        ("MultiplicativeScatterCorrection", pp.MultiplicativeScatterCorrection()),
        ("Derivate", pp.Derivate()),
        ("Gaussian", pp.Gaussian()),
        ("Wavelet", pp.Wavelet()),
        ("Haar", pp.Haar()),
        ("SimpleScale", pp.SimpleScale()),
    ]
    return preprocessings


@pytest.fixture(scope="session")
def augmentation_source_data():
    csv_aug = np.loadtxt(
        path_to("test_augmentation.csv"), delimiter=";")
    return csv_aug[:, 1:]


# @pytest.fixture(scope="session")
# def augmenters():
#     seed = 42
#     return [
#         (0, "Rotate_Translate", aug.Rotate_Translate, {"random_state": seed}),
#         (1, "Rotate_Translate_custom", aug.Rotate_Translate, {"random_state": seed, "p_range": 5, "y_factor": 5}),
#         (2, "Random_X_Operation", aug.Random_X_Operation, {"random_state": seed}),
#         (3, "Random_X_Operation_custom", aug.Random_X_Operation, {"random_state": seed, "operator_func": operator.add, "operator_range": (-0.002, 0.002)}),
#         (4, "Spline_Smoothing", aug.Spline_Smoothing, {"random_state": seed}),
#         (5, "Spline_X_Perturbation", aug.Spline_X_Perturbations, {"random_state": seed}),
#         (6, "Spline_X_Perturbation", aug.Spline_X_Perturbations, {"random_state": seed, "perturbation_density": 0.01, "perturbation_range": (-30, 30)}),
#         (7, "Spline_Y_Perturbation", aug.Spline_Y_Perturbations, {"random_state": seed}),
#         (8, "Spline_Y_Perturbation", aug.Spline_Y_Perturbations, {"random_state": seed, "spline_points": 5, "perturbation_intensity": 0.02}),
#         (9, "Spline_X_Simplification", aug.Spline_X_Simplification, {"random_state": seed}),
#         (10, "Spline_X_Simplification", aug.Spline_X_Simplification, {"random_state": seed, "spline_points": 15}),
#         (11, "Spline_Curve_Simplification", aug.Spline_Curve_Simplification, {"random_state": seed}),
#         (12, "Spline_Curve_Simplification", aug.Spline_Curve_Simplification, {"random_state": seed, "spline_points": 5, "uniform": False}),
#         (13, "Spline_Curve_Simplification", aug.Spline_Curve_Simplification, {"random_state": seed, "spline_points": 5, "uniform": True}),
#     ]

# Fixture for loading augmented data from CSV files


@pytest.fixture(scope="function")
def augmented_global_data_csv(index, name):
    data = np.loadtxt(
        path_to(f"test_augmentation_{index}_{name}_global.csv"), delimiter=";")
    return data


@pytest.fixture(scope="function")
def augmented_samples_data_csv(index, name):
    data = np.loadtxt(
        path_to(f"test_augmentation_{index}_{name}_samples.csv"), delimiter=";")
    return data
