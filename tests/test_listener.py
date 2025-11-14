import os
import numpy as np
import pytest
from multiprocessing import Queue
from unittest.mock import MagicMock

from src.service.listener import Listener
from src.proto.mlflow_probs_pb2 import MlflowProbs, PredictionList


@pytest.fixture
def mock_middleware():
    m = MagicMock()
    m.create_channel.return_value = "fake_channel"
    return m


@pytest.fixture
def mock_config():
    class C:
        num_workers = 2
    return C()


def build_message(client_id="clientA", batch_index=0, n_preds=3):
    msg = MlflowProbs()
    msg.client_id = client_id
    msg.batch_index = batch_index
    for _ in range(n_preds):
        pred = PredictionList()
        pred.values.extend([0.1, 0.2, 0.7])
        msg.pred.append(pred)
    return msg.SerializeToString()


def test_start_worker_pool(monkeypatch, mock_middleware, mock_config):
    # Patch Process so no real child process is created
    class FakeProcess:
        def __init__(self, target, args):
            self.started = False
        def start(self):
            self.started = True

    monkeypatch.setattr("src.service.listener.Process", FakeProcess)

    listener = Listener(mock_middleware, mock_config)
    listener.start_worker_pool()

    assert len(listener._active_workers) == mock_config.num_workers
    assert all(p.started for p in listener._active_workers)


def test_on_message_creates_dirs_and_enqueues(monkeypatch, tmp_path, mock_middleware, mock_config):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    monkeypatch.setattr("os.listdir", lambda p: [])
    monkeypatch.setattr("os.makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr("os.path.join", lambda *args: "/".join(args))

    listener = Listener(mock_middleware, mock_config)

    msg_bytes = build_message(client_id="user123", batch_index=7)
    listener._on_message(msg_bytes)

    client_id, session_id, batch_index, probs = listener._workers_queue.get()
    assert client_id == "user123"
    assert batch_index == 7
    assert isinstance(probs, np.ndarray)
    assert probs.shape[1] == 3  # the pred vector size


def test_on_message_error(monkeypatch, mock_middleware, mock_config):
    listener = Listener(mock_middleware, mock_config)
    
    with pytest.raises(Exception):
        listener._on_message(b"corrupted_message_bytes")


def test_handle_sigterm(monkeypatch, mock_middleware, mock_config):
    class FakeWorker:
        def __init__(self):
            self.joined = False
        def join(self):
            self.joined = True

    listener = Listener(mock_middleware, mock_config)
    listener._active_workers = [FakeWorker(), FakeWorker()]
    listener._workers_queue = Queue()

    listener.handle_sigterm(None, None)

    # Queue should get termination signals
    term1 = listener._workers_queue.get()
    term2 = listener._workers_queue.get()
    assert term1 == (None, None, None, None)
    assert term2 == (None, None, None, None)

    # Workers joined
    assert all(w.joined for w in listener._active_workers)
