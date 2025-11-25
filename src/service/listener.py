
import http
import logging
from multiprocessing import Queue
import uuid
from src.service.run_registry import RunRegistry
from src.lib.config import ARTIFACTS_DIR, MLFLOW_QUEUE_NAME
from multiprocessing import Process
from src.service.mlflow_logger import MlflowLogger
from src.proto import mlflow_probs_pb2
import os
import numpy as np
import mlflow

class Listener:
    def __init__(self, middleware, config, db):
        self._config = config
        self._middleware = middleware
        self._channel = self._middleware.create_channel()
        self._workers_queue = Queue()
        self._active_workers = []
        self._run_registry = db
        self._session_id = str(uuid.uuid4())  # Placeholder for session ID retrieval

    def run(self):
        self.start_worker_pool()
        self._middleware.basic_consume(
            self._channel, MLFLOW_QUEUE_NAME, self._on_message
        )
        self._middleware.start_consuming(self._channel)
        
        for worker in self._active_workers:
            worker.join() 

    def start_worker_pool(self):
        for i in range(self._config.num_workers):
            mlflow_logger = MlflowLogger(self._workers_queue, self._config.tracking_uri)
            mlflow_logger.start()
            logging.info(f"Started MlflowLogger worker process with PID {mlflow_logger.pid}")
            self._active_workers.append(mlflow_logger)  
    
    def handle_sigterm(self, signum, frame):
        """Handle SIGTERM signal for graceful shutdown."""
        self._middleware.stop_consuming(self._channel)
        for _ in self._active_workers:
            self._workers_queue.put((None, None))
        for worker in self._active_workers:
            worker.join() 


    def _on_message(self, ch, method, properties, body):
        try:
            logging.info("Received message for MLflow logging")
            message = mlflow_probs_pb2.MlflowProbs()
            message.ParseFromString(body)
            session_id = message.session_id

            run_id = self._run_registry.get_run_id(session_id)

            if run_id is None:
                mlflow.set_tracking_uri(self._config.tracking_uri)
                mlflow.set_experiment(message.client_id)

                with mlflow.start_run(run_name=session_id) as run:
                    run_id = run.info.run_id

                self._run_registry.save_run_id(session_id, run_id)

            self._workers_queue.put(
                (message, run_id)
            )

        except Exception as e:
            logging.exception("Unhandled exception in _on_message")
            raise e



       
        



