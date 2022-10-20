from os.path import abspath
from os.path import dirname as d

import numpy as np
import pytest

from pinard import preprocessing as pp

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
    return split_validation_data.astype(np.int32)


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
