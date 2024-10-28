import pytest
from unittest.mock import MagicMock, patch
from pinard.core.finetuner import (
    BaseFineTuner,
    OptunaFineTuner,
    SklearnFineTuner,
    FineTunerFactory
)


def test_base_finetuner():
    with pytest.raises(TypeError):
        BaseFineTuner(model_manager=None)


def test_optuna_finetuner_initialization():
    model_manager = MagicMock()
    finetuner = OptunaFineTuner(model_manager)
    assert finetuner.model_manager == model_manager
    assert finetuner.model_config == model_manager.model_config


def test_sklearn_finetuner_initialization():
    model_manager = MagicMock()
    finetuner = SklearnFineTuner(model_manager)
    assert finetuner.model_manager == model_manager
    assert finetuner.model_config == model_manager.model_config


def test_finetuner_factory_optuna():
    model_manager = MagicMock()
    finetuner = FineTunerFactory.get_fine_tuner('optuna', model_manager)
    assert isinstance(finetuner, OptunaFineTuner)


def test_finetuner_factory_sklearn():
    model_manager = MagicMock()
    finetuner = FineTunerFactory.get_fine_tuner('sklearn', model_manager)
    assert isinstance(finetuner, SklearnFineTuner)


def test_finetuner_factory_invalid():
    model_manager = MagicMock()
    with pytest.raises(ValueError):
        FineTunerFactory.get_fine_tuner('invalid', model_manager)


@patch('pinard.core.finetuner.optuna.create_study')
def test_optuna_finetuner_finetune(mock_create_study):
    model_manager = MagicMock()
    finetuner = OptunaFineTuner(model_manager)
    dataset = MagicMock()
    finetune_params = {'model_params': {}, 'training_params': {}}
    finetuner.finetune(dataset, finetune_params)
    mock_create_study.assert_called_once()


@patch('pinard.core.finetuner.GridSearchCV')
def test_sklearn_finetuner_finetune(mock_grid_search_cv):
    model_manager = MagicMock()
    finetuner = SklearnFineTuner(model_manager)
    dataset = MagicMock()
    finetune_params = {'model_params': {}}
    finetuner.finetune(dataset, finetune_params)
    mock_grid_search_cv.assert_called_once()
