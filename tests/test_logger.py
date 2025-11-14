import os
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from src.service.mlflow_logger import MlflowLogger


class FakeQueue:
    def __init__(self, items):
        self.items = list(items)
    def get(self):
        return self.items.pop(0)


def test_log_single_batch_creates_artifact(monkeypatch, tmp_path):
    logger = MlflowLogger(None)
    logger._curr_client = "clientA"
    logger._curr_session = "sessionB"

    artifacts = tmp_path / "artifacts" / "clientA" / "sessionB"
    artifacts.mkdir(parents=True)

    monkeypatch.setattr("os.path.join", lambda *args: "/".join(args))
    monkeypatch.setattr("src.service.mlflow_logger.os.remove", lambda x: None)

    fake_log_artifact = MagicMock()
    monkeypatch.setattr("mlflow.log_artifact", fake_log_artifact)

    inputs = np.random.rand(3, 1, 28, 28).astype(np.float32)
    probs = np.random.rand(3, 10).astype(np.float32)
    labels = [1, 2, 3]

    monkeypatch.setattr("pandas.DataFrame.to_parquet", lambda self, path, index: None)

    logger.log_single_batch(batch_index=0, probs=probs, inputs=inputs, labels=labels)

    fake_log_artifact.assert_called()


def test_run_processes_items(monkeypatch):
    # One real item + termination
    queue = FakeQueue([
        ("clientA", "session1", 0, np.random.rand(2, 10).astype(np.float32)),
        (None, None, None, None)
    ])

    logger = MlflowLogger(queue)

    # Patch heavy calls
    monkeypatch.setattr("mlflow.set_experiment", lambda x: None)
    monkeypatch.setattr("mlflow.start_run", lambda run_id=None: None)
    monkeypatch.setattr("mlflow.end_run", lambda : None)
    monkeypatch.setattr("src.service.mlflow_logger.MlflowLogger.log_single_batch", lambda *args, **kwargs: None)

    # Run expects to be executed (we call run() directly, not start())
    logger.run()

    # Should have processed first item and exited cleanly
