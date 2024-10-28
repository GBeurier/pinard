import pytest
from unittest.mock import MagicMock
from pinard.core.utils import framework, get_full_import_path


def test_framework_decorator():
    @framework('tensorflow')
    def dummy_function():
        pass

    assert dummy_function.framework == 'tensorflow'


def test_get_full_import_path():
    class DummyClass:
        pass

    instance = DummyClass()
    path = get_full_import_path(instance)
    assert path == 'test_utils.DummyClass'
