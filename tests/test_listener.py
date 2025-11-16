import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from src.service.listener import Listener
from src.proto import mlflow_probs_pb2

@pytest.fixture
def dummy_config():
    cfg = MagicMock()
    cfg.num_workers = 1
    cfg.tracking_uri = "http://mlflow:5009"
    return cfg

@pytest.fixture
def dummy_db():
    db = MagicMock()
    db.get_run_id.return_value = None
    return db

@pytest.fixture
def dummy_middleware():
    m = MagicMock()
    m.create_channel.return_value = MagicMock()
    return m

def _build_message():
    msg = mlflow_probs_pb2.MlflowProbs()
    msg.client_id = "123"
    msg.batch_index = 4

    p = msg.pred.add()
    p.values[:] = [0.1, 0.2, 0.3]

    return msg

@patch("mlflow.start_run")
@patch("mlflow.set_experiment")
@patch("mlflow.set_tracking_uri")
def test_on_message_creates_run(
    mock_set_uri, mock_set_exp, mock_start_run, dummy_config, dummy_middleware, dummy_db, tmp_path
):
    # Setup
    with patch("src.service.listener.ARTIFACTS_DIR", str(tmp_path)):
        listener = Listener(dummy_middleware, dummy_config, dummy_db)
        listener._workers_queue = MagicMock()

        message = _build_message()
        body = message.SerializeToString()

        run_mock = MagicMock()
        run_mock.info.run_id = "run_generated"
        mock_start_run.return_value.__enter__.return_value = run_mock

        listener._on_message(None, None, None, body)

        dummy_db.save_run_id.assert_called_once_with("session_123", "run_generated")
        listener._workers_queue.put.assert_called_once()
        args = listener._workers_queue.put.call_args[0][0]

        assert args[0] == "123"          
        assert args[1] == "run_generated"
        assert args[2] == "session_123"
        assert args[3] == 4             
def test_on_message_uses_existing_run(dummy_config, dummy_middleware, tmp_path):
    dummy_db = MagicMock()
    dummy_db.get_run_id.return_value = "existing_run"

    with patch("src.service.listener.ARTIFACTS_DIR", str(tmp_path)):
        listener = Listener(dummy_middleware, dummy_config, dummy_db)
        listener._workers_queue = MagicMock()

        msg = _build_message()
        body = msg.SerializeToString()

        with patch("mlflow.start_run") as start_mock:
            listener._on_message(None, None, None, body)

            start_mock.assert_not_called()

            listener._workers_queue.put.assert_called_once()
            args = listener._workers_queue.put.call_args[0][0]
            assert args[1] == "existing_run"
