from pinard import utils


def test_single_file_load(simple_data):
    x, y = utils.load_csv(simple_data, y_cols=[0, 1], x_hdr=0, x_index_col=[0, 1])
    assert x.shape == (6, 3)
    assert x[0, 1] == 2


def test_single_file_load_na(simple_data_na):
    x, y = utils.load_csv(simple_data_na, y_cols=[0, 1], x_hdr=0, x_index_col=[0, 1], remove_na=True)
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
            simple_data,
            simple_data_na,
            y_cols=[0, 1],
            x_hdr=0,
            y_hdr=0,
            x_index_col=[0, 1, 2, 3],
            y_index_col=[0, 1],
            remove_na=True
        )
    except utils.WrongFormatError as error:
        assert error.x.shape == (6, 3)
        assert error.y.shape == (4, 2)
