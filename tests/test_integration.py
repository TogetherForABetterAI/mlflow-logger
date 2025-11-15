import os
from unittest.mock import Mock, patch
from src.lib.logger import initialize_logging
from src.service.listener import Listener
import pytest
from tests.mocks.middleware import FakeMiddleware

@pytest.fixture
def mock_config():
    config = Mock(
        tracking_uri="http://mlflow:5000",
        logging_level="INFO",
        host="rabbitmq",
        port=1234,
        username="rabbit",
        password="rabbit",
        max_retries=3,
        num_workers=4,
    )
    
    return config

# @patch('src.service.mlflow_logger.mlflow.set_experiment')
# @patch('src.service.mlflow_logger.mlflow.start_run')
# @patch('src.service.mlflow_logger.mlflow.end_run')
def test_integration(mock_config):
    mock_middleware = FakeMiddleware(mock_config)
    initialize_logging(mock_config.logging_level)
    listener = Listener(mock_middleware, mock_config)
    listener.run()


    