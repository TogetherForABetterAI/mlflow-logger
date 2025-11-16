import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from src.service.mlflow_logger import MlflowLogger

@pytest.fixture
def logger(tmp_path):
    q = MagicMock()
    ml = MlflowLogger(q, "http://mlflow:5009")
    ml._curr_client = "c1"
    ml._curr_session = "s1"
    ml._curr_run_id = "run1"
    return ml


@patch("mlflow.set_tracking_uri")
@patch("mlflow.start_run")
@patch("mlflow.end_run")
@patch("mlflow.set_experiment")
def test_run_main_loop(mock_exp, mock_end, mock_start, mock_set, tmp_path):
    q = MagicMock()
    logger = MlflowLogger(q, "http://mlflow:5009")

    # Simula mensajes en la queue
    q.get.side_effect = [
        ("c1", "run1", "session1", 0, np.zeros((2, 10))),
        (None, None, None, None, None),   # señal de terminar
    ]

    # Evita escribir parquet durante esta prueba
    with patch.object(logger, "log_single_batch") as log_mock:
        logger.run()

        assert log_mock.call_count == 1
        mock_start.assert_called_once()
        mock_end.assert_called_once()
