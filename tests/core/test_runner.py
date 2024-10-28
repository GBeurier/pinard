import pytest
from unittest.mock import MagicMock, patch
from pinard.core.runner import ExperimentRunner


@patch('pinard.core.runner.ExperimentManager')
def test_experiment_runner_initialization(mock_manager):
    configs = MagicMock()
    runner = ExperimentRunner(configs)
    assert runner.configs == configs
    assert runner.manager == mock_manager.return_value


# @patch('pinard.core.runner.get_dataset')
# @patch('pinard.core.runner.run_pipeline')
# @patch('pinard.core.runner.ModelManagerFactory.get_model_manager')
# def test_experiment_runner_run(mock_get_model_manager, mock_run_pipeline, mock_get_dataset):
#     configs = MagicMock()
#     config = MagicMock()
#     configs.__iter__.return_value = [config]
#     runner = ExperimentRunner(configs)
#     runner.run()
#     mock_get_dataset.assert_called_once()
#     mock_run_pipeline.assert_called_once()
#     mock_get_model_manager.assert_called_once()
