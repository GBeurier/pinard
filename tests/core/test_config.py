import pytest
from pinard.core.config import Config


def test_config_initialization():
    config = Config(
        dataset='my_dataset',
        x_pipeline='my_x_pipeline',
        y_pipeline='my_y_pipeline',
        model='my_model',
        experiment={'param': 'value'},
        seed=42
    )
    assert config.dataset == 'my_dataset'
    assert config.x_pipeline == 'my_x_pipeline'
    assert config.y_pipeline == 'my_y_pipeline'
    assert config.model == 'my_model'
    assert config.experiment == {'param': 'value'}
    assert config.seed == 42


def test_config_defaults():
    config = Config(dataset='my_dataset')
    assert config.dataset == 'my_dataset'
    assert config.x_pipeline is None
    assert config.y_pipeline is None
    assert config.model is None
    assert config.experiment is None
    assert config.seed is None
