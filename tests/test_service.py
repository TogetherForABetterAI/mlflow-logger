# tests/test_listener_mlflow.py
import os
import io
import types
import logging
import multiprocessing
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import importlib

class FakePred:
    def __init__(self, values):
        self.values = values


class FakeMlflowProbs:
    """
    Fake "module" que contiene la clase MlflowProbs con ParseFromString.
    La ParseFromString llenará atributos: client_id, pred (lista de FakePred), batch_index.
    """
    class MlflowProbs:
        def __init__(self):
            self.client_id = None
            self.pred = []
            self.batch_index = None

        def ParseFromString(self, body: bytes):
            try:
                s = body.decode()
                parts = s.split(":")
                self.client_id = parts[0]
                self.batch_index = int(parts[1])
                preds = []
                for p in parts[2].split("|"):
                    vals = [float(x) for x in p.split(",") if x != ""]
                    preds.append(FakePred(vals))
                self.pred = preds
            except Exception:
                pass

@pytest.fixture(autouse=True)
def disable_logging(monkeypatch):
    """Silencia logs durante tests."""
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: None)
    yield


@pytest.fixture
def tmp_artifacts_dir(tmp_path, monkeypatch):
    """Crea un ARTIFACTS_DIR temporal y parchea los módulos que lo importaron."""
    art_dir = tmp_path / "artifacts"
    art_dir.mkdir()
    monkeypatch.setenv("ARTIFACTS_PATH", str(art_dir))  # por si tu código lee env
    monkeypatch.setattr("src.service.listener.ARTIFACTS_DIR", str(art_dir), raising=False)
    monkeypatch.setattr("src.service.mlflow_logger.ARTIFACTS_DIR", str(art_dir), raising=False)
    yield str(art_dir)


@pytest.fixture
def fake_mlflow(monkeypatch):
    """
    Provee un fake mlflow con:
      - set_tracking_uri
      - set_experiment
      - start_run: soporta "with mlflow.start_run(...)" y devuelve objeto con info.run_id
      - log_artifact (Mock)
    """
    fake = types.SimpleNamespace()
    fake.set_tracking_uri = Mock()
    fake.set_experiment = Mock()

    class RunCtx:
        def __init__(self, run_id):
            self.info = SimpleNamespace(run_id=run_id)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    counter = {"i": 0}

    def start_run(**kwargs):
        if "run_name" in kwargs and "run_id" not in kwargs:
            counter["i"] += 1
            rid = f"run_{counter['i']:04d}"
            return RunCtx(rid)
        elif "run_id" in kwargs:
            return RunCtx(kwargs["run_id"])
        else:
            counter["i"] += 1
            rid = f"run_{counter['i']:04d}"
            return RunCtx(rid)

    fake.start_run = Mock(side_effect=start_run)
    fake.log_artifact = Mock()
    fake.end_run = Mock()

    monkeypatch.setattr("src.service.listener.mlflow", fake, raising=False)
    monkeypatch.setattr("src.service.mlflow_logger.mlflow", fake, raising=False)

    yield fake


class SynchronousFakeMiddleware:
    def __init__(self):
        self._callback = None
        self._channel = object()

    def create_channel(self):
        return self._channel

    def basic_consume(self, channel, queue_name, callback):
        self._callback = callback

    def start_consuming(self, channel):
        pass

    def stop_consuming(self, channel):
        pass

def test_on_message_creates_run_and_enqueue(tmp_artifacts_dir, monkeypatch, fake_mlflow):
    """
    Si no existe run_id.txt, Listener._on_message debe:
      - crear un experiment (llamando a mlflow.set_experiment)
      - crear una run usando mlflow.start_run(run_name=...)
      - escribir run_id.txt
      - encolar (client_id, run_id, session_id, batch_index, probs)
    """
    from src.service.listener import Listener

    monkeypatch.setattr("src.service.listener.mlflow_probs_pb2", FakeMlflowProbs, raising=False)

    cfg = SimpleNamespace(tracking_uri="http://mlflow:5009", num_workers=1, logging_level="INFO")

    mw = SynchronousFakeMiddleware()
    listener = Listener(mw, cfg)

    body = b"clientA:1:0.1,0.9|0.2,0.8"  # clientA, batch 1, two preds
    listener._on_message(None, None, None, body)

    session_id = listener._get_session_id_from_client("clientA")
    run_file = os.path.join(tmp_artifacts_dir, "clientA", session_id, "run_id.txt")
    assert os.path.exists(run_file), "run_id.txt should exist after handling first message"

    with open(run_file, "r") as f:
        run_id = f.read().strip()
    assert run_id != "", "run_id must be non empty"

    fake_mlflow.set_tracking_uri.assert_called()
    fake_mlflow.set_experiment.assert_called_with("Calibration_Client_clientA")
    assert fake_mlflow.start_run.call_count >= 1

    q = listener._workers_queue
    item = q.get(timeout=1)
    assert item[0] == "clientA"
    assert item[1] == run_id
    assert isinstance(item[4], np.ndarray)
    q.get_nowait() if not q.empty() else None


def test_on_message_uses_existing_run_id(tmp_artifacts_dir, monkeypatch, fake_mlflow):
    """
    Si run_id.txt existe, no se debe crear una nueva run; se debe leer el run_id y encolar.
    """
    from src.service.listener import Listener
    monkeypatch.setattr("src.service.listener.mlflow_probs_pb2", FakeMlflowProbs, raising=False)

    cfg = SimpleNamespace(tracking_uri="http://mlflow:5009", num_workers=1, logging_level="INFO")
    mw = SynchronousFakeMiddleware()
    listener = Listener(mw, cfg)

    session_id = listener._get_session_id_from_client("clientB")
    client_dir = os.path.join(tmp_artifacts_dir, "clientB")
    os.makedirs(os.path.join(client_dir, session_id), exist_ok=True)
    known_run_id = "existing_run_0001"
    run_file = os.path.join(client_dir, session_id, "run_id.txt")
    with open(run_file, "w") as f:
        f.write(known_run_id)

    body = b"clientB:2:0.1,0.9"
    listener._on_message(None, None, None, body)


    q = listener._workers_queue
    item = q.get(timeout=1)
    assert item[0] == "clientB"
    assert item[1] == known_run_id
    assert item[2] == session_id
    assert isinstance(item[4], np.ndarray)


def test_mlflowlogger_logs_artifact_and_zero_padding(tmp_artifacts_dir, monkeypatch, fake_mlflow):
    """
    Test that MlflowLogger.run() writes a parquet file with zero-padded name,
    calls mlflow.log_artifact with that path, and then removes the file.
    """
    from src.service.mlflow_logger import MlflowLogger

    q = multiprocessing.Queue()
    client = "clientX"
    session = "session_clientX"
    batch_index = 1
    probs = np.array([[0.1, 0.9], [0.2, 0.8]], dtype=np.float32)

    target_dir = os.path.join(tmp_artifacts_dir, client, session)
    os.makedirs(target_dir, exist_ok=True)

    q.put((client, "run_0001", session, batch_index, probs))
    q.put((None, None, None, None, None))  # worker will detect probs is None and break

    ml_logger = MlflowLogger(q, "http://mlflow:5009")

    ml_logger.run()

    expected_filename = f"batch_{batch_index:05d}.parquet"
    expected_path = os.path.join(tmp_artifacts_dir, client, session, expected_filename)

    assert not os.path.exists(expected_path), "Artifact file should be removed after log_artifact"

    called_paths = [args[0][0] for args in fake_mlflow.log_artifact.call_args_list]
    assert any(p.endswith(expected_filename) for p in called_paths), f"mlflow.log_artifact should be called with *{expected_filename}*; got {called_paths}"
