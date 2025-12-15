import numpy as np
import pandas as pd
import mlflow
import os
import logging
from typing import List
from multiprocessing import Process

from src.lib.config import ARTIFACTS_DIR


class MlflowLogger(Process):
    def __init__(
        self,
        workers_queue,
        ack_queue,
        tracking_uri,
        tracking_username,
        tracking_password,
    ):
        super().__init__()
        self.logger = logging.getLogger(f"mlflow-logger-{self.pid}")
        self.logger.info("MlflowLogger process started")
        self._workers_queue = workers_queue
        self._ack_queue = ack_queue
        self._tracking_uri = tracking_uri
        self._tracking_username = tracking_username
        self._tracking_password = tracking_password

    def run(self):
        os.environ["MLFLOW_TRACKING_USERNAME"] = self._tracking_username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = self._tracking_password

        while True:
            mlflow_data = self._workers_queue.get()
            if mlflow_data is None:
                break

            client_id = mlflow_data.client_id
            batch_index = mlflow_data.batch_index
            inputs = mlflow_data.inputs
            labels = mlflow_data.labels
            delivery_tag = mlflow_data.delivery_tag

            success = True
            try:
                mlflow.set_tracking_uri(self._tracking_uri)

                self._experiment = mlflow.set_experiment(client_id)

                mlflow.start_run(run_id=mlflow_data.run_id)
                self.log_single_batch(batch_index, mlflow_data.pred, inputs, labels)

                logging.info(
                    f"[MLflow] Logged batch {batch_index} for client {client_id}"
                )
                mlflow.end_run()
            except Exception as e:
                logging.error(f"[MLflow] Error logging batch {batch_index}: {e}")
                success = False

            # send confirmation back to ack_thread
            if delivery_tag is not None:
                self._ack_queue.put((delivery_tag, success))

    def log_single_batch(
        self, batch_index: int, probs: np.ndarray, inputs: np.ndarray, labels: List[int]
    ):
        probs_list = probs.tolist()

        n_samples = len(labels)
        total_bytes = len(inputs)
        img_byte_size = total_bytes // n_samples
        inputs_split = [
            inputs[i * img_byte_size : (i + 1) * img_byte_size]
            for i in range(n_samples)
        ]

        df = pd.DataFrame(
            {"input": inputs_split, "y_pred": probs_list, "y_test": labels}
        )

        os.makedirs(ARTIFACTS_DIR, exist_ok=True)

        filename = f"batch_{batch_index:05d}.parquet"
        file_path = os.path.join(f"{ARTIFACTS_DIR}", filename)
        df.to_parquet(file_path, index=False)

        mlflow.log_artifact(file_path)
        os.remove(file_path)
