
from multiprocessing import Queue
from src.lib.config import MLFLOW_QUEUE_NAME
from multiprocessing import Process
from src.service.mlflow_logger import MlflowLogger
from proto import mlflow_pb2
import os
import numpy as np


class Listener:
    def __init__(self, middleware, config):
        self._config = config
        self._middleware = middleware
        self._channel = self._middleware.create_channel()
        self._workers_queue = Queue()
        self._clients = {}
        for client_id in os.listdir("/artifacts"):
            for session_id in os.listdir(os.path.join("/artifacts", client_id)):
                self._clients[client_id] = session_id
                
        self._active_workers = []


    def run(self):
        self.start_worker_pool()
        self._middleware.basic_consume(
            self._channel, MLFLOW_QUEUE_NAME, self.on_message
        )
        self._middleware.start_consuming(self._channel)

    def start_worker_pool(self):
        for i in range(self._config.num_workers):
            p = Process(target=MlflowLogger, args=(self._workers_queue,))
            p.start()
            self._active_workers.append(p)  

    def _get_session_id_from_client(self, client_id: str) -> str:
        """Retrieve the session ID for a given client ID."""
        # Placeholder implementation; replace with actual logic to retrieve session ID
        return "session_" + client_id
    
    def handle_sigterm(self, signum, frame):
        """Handle SIGTERM signal for graceful shutdown."""
        self._middleware.stop_consuming(self._channel)
        for _ in self._active_workers:
            self._workers_queue.put((None, None, None, None))
        for worker in self._active_workers:
            worker.join() 

    def _handle_probability_message(self, body):
        """Handle probability messages from the calibration queue."""

        try:
            message = mlflow_pb2.MlflowMessage()
            message.ParseFromString(body)

            probs = np.array([list(p.values) for p in message.pred], dtype=np.float32)

            # Check if the client_id dir is created in artifacts
            if message.client_id not in self._clients:
                os.makedirs(os.path.join("/artifacts", message.client_id), exist_ok=True)
                self._clients.add(message.client_id)

            if session_id not in self._clients[message.client_id]:
                os.makedirs(os.path.join("/artifacts", message.client_id, session_id), exist_ok=True)
                self._clients[message.client_id] = session_id

            session_id = self._get_session_id_from_client(message.client_id)
            self._workers_queue.put((message.client_id, session_id, message.batch_index, probs))
            
        except Exception as e:
            raise e



       
        



