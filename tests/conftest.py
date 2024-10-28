
import pytest
import numpy as np

@pytest.fixture(scope="module")
def simple_data():
    """Fixture to generate simple synthetic data for testing."""
    return np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                     [6.0, 7.0, 8.0, 9.0, 10.0]])


@pytest.fixture(scope="module")
def random_data():
    """Fixture to generate random synthetic data for testing."""
    return np.random.rand(10, 100)



import pytest
import numpy as np

@pytest.fixture(scope="module")
def simple_data():
    """Fixture to generate simple synthetic data for testing."""
    return np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                     [6.0, 7.0, 8.0, 9.0, 10.0]])


@pytest.fixture(scope="module")
def random_data():
    """Fixture to generate random synthetic data for testing."""
    return np.random.rand(10, 100)


