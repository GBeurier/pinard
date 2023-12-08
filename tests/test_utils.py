from pinard import utils
import pytest


def test_single_file_load(simple_data):
    x, y = utils.load_csv(simple_data, y_cols=[0, 1], x_hdr=0, x_index_col=[0, 1])
    assert x.shape == (6, 3)
    assert x[0, 1] == 2


def test_single_file_load_na(simple_data_na):
    x, y = utils.load_csv(simple_data_na, y_cols=[0, 1], x_hdr=0, x_index_col=[0, 1], autoremove_na=True)
    assert x.shape == (4, 3)
    assert x[0, 1] == 2


def test_double_file_load(simple_data, simple_data_na):
    x, y = utils.load_csv(
        simple_data,
        simple_data,
        y_cols=[0, 1],
        x_hdr=0,
        y_hdr=0,
        x_index_col=[0, 1, 2, 3],
        y_index_col=[0, 1],
    )
    assert x.shape == (6, 3)
    assert y.shape == (6, 2)
    assert x[0, 1] == 2
    assert y[0, 1] == 2


def test_double_file_load_na(simple_data, simple_data_na):
    try:
        x, y = utils.load_csv(
            simple_data, simple_data_na, y_cols=[0, 1], x_hdr=0, y_hdr=0, x_index_col=[0, 1, 2, 3], y_index_col=[0, 1], autoremove_na=True
        )
    except utils.WrongFormatError as error:
        assert error.x.shape == (6, 3)
        assert error.y.shape == (4, 2)


def test_errors(bad_data):
    try:
        utils.load_csv(bad_data)
    except Exception as e:
        assert isinstance(e, utils.WrongFormatError)


# def test_invalid_x_shape(bad_data):
#     with pytest.raises(utils.WrongFormatError) as excinfo:
#         utils.load_csv(bad_data)
#     assert "Invalid X shape:" in str(excinfo.value)


# def test_invalid_y_shape(bad_data):
#     with pytest.raises(utils.WrongFormatError) as excinfo:
#         x, y = utils.load_csv(bad_data)
#     assert "Invalid Y shape:" in str(excinfo.value)


# def test_invalid_both_shapes(bad_data):
#     with pytest.raises(utils.WrongFormatError) as excinfo:
#         x, y = utils.load_csv(bad_data)
#     exc_message = str(excinfo.value)
#     assert "Invalid X shape:" in exc_message
#     assert "Invalid Y shape:" in exc_message


# def test_analyze_csv_file():
#     filename = 'test_data.csv'  # Replace with a path to a test CSV file
#     delimiter, numeric_delimiter, header = analyze_csv_file(filename)

#     assert delimiter in [';', '\t', ','], "Delimiter not correctly identified"
#     assert numeric_delimiter in ['.', ','], "Numeric delimiter not correctly identified"
#     assert isinstance(header, (int, type(None))), "Header detection failed"


# def test_load_csv():
#     file_path = 'test_data.csv'  # Replace with a path to a test CSV file
#     data, indexes_with_na = load_csv(file_path, na_policy='auto')

#     assert isinstance(data, np.ndarray), "Data is not in numpy array format"
#     assert all(not np.isnan(data).any()), "NA values not handled correctly"


# def test_load_XY_csv_single_file():
#     x_data, y_data = load_XY_csv('test_data.csv', y_cols=[0], sep=';', x_hdr=0)

#     assert isinstance(x_data, np.ndarray) and isinstance(y_data, np.ndarray), "Data is not in numpy array format"
#     assert len(x_data) == len(y_data), "X and Y datasets have different row counts"


# def test_load_XY_csv_separate_files():
#     x_data, y_data = load_XY_csv('test_x_data.csv', y_fname='test_y_data.csv', sep=';', x_hdr=0, y_hdr=0)

#     assert isinstance(x_data, np.ndarray) and isinstance(y_data, np.ndarray), "Data is not in numpy array format"
#     assert len(x_data) == len(y_data), "X and Y datasets have different row counts"


# def test_load_XY_csv_error_handling():
#     with pytest.raises(ValueError):
#         load_XY_csv('test_data.csv', y_fname=None, y_cols=None, sep=';', x_hdr=0)
