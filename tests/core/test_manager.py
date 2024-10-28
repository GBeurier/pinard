import pytest
from unittest.mock import MagicMock, patch
from pinard.core.manager import ExperimentManager


def test_experiment_manager_initialization():
    manager = ExperimentManager(results_dir='results', resume_mode='skip', verbose=1)
    assert manager.results_dir == 'results'
    assert manager.resume_mode == 'skip'
    assert manager.verbose == 1
    assert manager.experiment_info == {}
    assert manager.experiment_path is None


# @patch('os.makedirs')
# @patch('os.path.exists', return_value=False)
# def test_prepare_experiment(mock_exists, mock_makedirs):
#     manager = ExperimentManager(results_dir='results')
#     config = MagicMock()
#     config.dataset = 'my_dataset'
#     config.model = {'class': 'my_model'}
#     config.seed = 42
#     manager.prepare_experiment(config)
#     assert manager.experiment_info['dataset_name'] == 'my_dataset'
#     assert manager.experiment_info['model_name'] == 'my_model'
#     assert manager.experiment_info['seed'] == 42


def test_make_config_serializable():
    manager = ExperimentManager(results_dir='results')
    config = MagicMock()
    config.dataset = 'my_dataset'
    config.model = {'class': 'my_model'}
    serializable_config = manager.make_config_serializable(config)
    assert serializable_config['dataset'] == 'my_dataset'
    assert serializable_config['model'] == {'class': 'my_model'}


# @patch('json.dump')
# def test_save_results(mock_json_dump):
#     manager = ExperimentManager(results_dir='results')
#     manager.experiment_path = 'experiment_path'
#     model_manager = MagicMock()
#     y_pred = [1, 2, 3]
#     y_true = [1, 2, 3]
#     metrics = ['mse']
#     manager.save_results(model_manager, y_pred, y_true, metrics)
#     mock_json_dump.assert_called()
