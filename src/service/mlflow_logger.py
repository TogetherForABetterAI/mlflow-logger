import numpy as np
import pandas as pd
import mlflow
import os
import logging
from typing import List
from multiprocessing import Process

from src.lib.config import ARTIFACTS_DIR


class MlflowLogger(Process):
    def __init__(self, workers_queue, tracking_uri):
        super().__init__()
        self.logger = logging.getLogger(f"mlflow-logger-{self.pid}")
        self.logger.info("MlflowLogger process started")
        self._workers_queue = workers_queue
        self._curr_client = None
        self._curr_session = None
        self._curr_run_id = None
        self._tracking_uri = tracking_uri

    def run(self):
        while True:
            client_id, run_id, session_id, batch_index, probs = self._workers_queue.get()
            if probs is None:
                break

            # TODO: Look for inputs data and labels
            # TODO: Look for client's session id

            batch_size = len(probs)
            mock_inputs = np.random.rand(batch_size, 1, 28, 28).astype(np.float32)
            mock_labels = np.random.randint(0, 10, size=batch_size).tolist()
            self._curr_client = client_id
            self._curr_run_id = run_id
            self._curr_session = session_id
            
            logging.info(f"[MLflow] Tracking URI: {self._tracking_uri}")
            mlflow.set_tracking_uri(self._tracking_uri)
            logging.info(f"[MLflow] Logging for client: {client_id}, session: {session_id}, run: {run_id}, batch: {batch_index}")
        
            self._experiment = mlflow.set_experiment("syngenta")

            mlflow.start_run(run_id=run_id) # TODO: Change to use client's session id
            self.log_single_batch(batch_index, probs, mock_inputs, mock_labels)
            mlflow.end_run()

    def log_single_batch(
        self, batch_index: int, probs: np.ndarray, inputs: np.ndarray, labels: List[int]
    ):
        input_flat = inputs.reshape(inputs.shape[0], -1).tolist()
        probs_list = probs.tolist()

        logging.info(
            f"[MLflow] Logging batch {batch_index} with {len(probs_list)} probabilities and {len(input_flat)} inputs"
        )

        df = pd.DataFrame({"input": input_flat, "y_pred": probs_list, "y_test": labels})

        filename = f"batch_{batch_index:05d}.parquet"
        file_path = os.path.join(f"{ARTIFACTS_DIR}", filename)
        df.to_parquet(file_path, index=False)
    
        mlflow.log_artifact(file_path)
        os.remove(file_path)

        logging.info(
            f"[MLflow] Logged batch {batch_index} for client {self._curr_client} to 'batches/{self._curr_client}/{filename}'"
        )