from fileinput import filename
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
        self._tracking_uri = tracking_uri

    def run(self):
        while True:
            message, run_id = self._workers_queue.get()
            if message is None:
                break

            probs = np.array([list(p.values) for p in message.pred], dtype=np.float32)            
            client_id = message.client_id
            batch_index = message.batch_index
            inputs = message.data
            labels = message.labels

            mlflow.set_tracking_uri(self._tracking_uri)
        
            self._experiment = mlflow.set_experiment(client_id)

            mlflow.start_run(run_id=run_id)
            self.log_single_batch(batch_index, probs, inputs, labels)
        
            logging.info(
                f"[MLflow] Logged batch {batch_index} for client {client_id}"
            )
            mlflow.end_run()

    def log_single_batch(
        self, batch_index: int, probs: np.ndarray, inputs: np.ndarray, labels: List[int]
    ):
        probs_list = probs.tolist()

        df = pd.DataFrame({"input": inputs, "y_pred": probs_list, "y_test": labels})

        filename = f"batch_{batch_index:05d}.parquet"
        file_path = os.path.join(f"{ARTIFACTS_DIR}", filename)
        df.to_parquet(file_path, index=False)
    
        mlflow.log_artifact(file_path)
        os.remove(file_path)

