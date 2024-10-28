import pytest
from unittest.mock import MagicMock, patch
from pinard.core.model_builder_factory import ModelBuilderFactory


def test_build_single_model_from_string():
    with patch('pinard.core.model_builder_factory.ModelBuilderFactory._from_string') as mock_from_string:
        ModelBuilderFactory.build_single_model('some.model.Class', None, None)
        mock_from_string.assert_called_once()


def test_build_single_model_from_dict():
    with patch('pinard.core.model_builder_factory.ModelBuilderFactory._from_dict') as mock_from_dict:
        model_config = {'class': 'some.model.Class'}
        ModelBuilderFactory.build_single_model(model_config, None, None)
        mock_from_dict.assert_called_once()


def test_build_single_model_from_instance():
    class DummyModel:
        pass

    model_instance = DummyModel()
    model = ModelBuilderFactory.build_single_model(model_instance, None, None)
    assert model == model_instance


# def test_build_single_model_invalid():
#     with pytest.raises(ValueError):
#         ModelBuilderFactory.build_single_model(None, None, None)
