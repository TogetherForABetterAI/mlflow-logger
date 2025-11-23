import os
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch
from src.service.run_registry import RunRegistry
from src.lib.logger import initialize_logging
from src.service.listener import Listener
import pytest
from tests.mocks.middleware import FakeMiddleware

@pytest.fixture
def mock_config():
    config = SimpleNamespace(
        tracking_uri="http://mlflow:5009",
        logging_level="INFO",
        host="rabbitmq",
        port=1234,
        username="rabbit",
        password="rabbit",
        max_retries=3,
        num_workers=4,
        db_uri="postgresql://user:password@test-sessions-db:5432/mlflow_db",
    )
    
    return config


def test_integration(mock_config):
    mock_middleware = FakeMiddleware(mock_config)
    initialize_logging(mock_config.logging_level)
    db = RunRegistry(mock_config.db_uri)
    listener = Listener(mock_middleware, mock_config, db)
    listener.run()
    

    