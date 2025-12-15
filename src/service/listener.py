import logging
from multiprocessing import Queue
import os
import uuid
import pika.exceptions
import threading

import numpy as np
from lib.model import LoggingDTO
from src.lib.config import MLFLOW_QUEUE_NAME
from src.service.mlflow_logger import MlflowLogger
from src.proto import mlflow_probs_pb2
import mlflow


class Listener:
    def __init__(self, middleware, config, db):
        self._config = config
        self._middleware = middleware
        self._channel = self._middleware.create_channel(
            prefetch_count=config.num_workers
        )
        self._workers_queue = Queue()
        self._ack_queue = Queue()  # Nueva queue para recibir confirmaciones
        self._active_workers = []
        self._run_registry = db
        self._session_id = str(uuid.uuid4())  # Placeholder for session ID retrieval
        self.tracking_username = config.mlflow_tracking_username
        self.tracking_password = config.mlflow_tracking_password
        self.shutdown_initiated = False

    def run(self):
        self.start_worker_pool()

        # Iniciar thread para procesar ACKs
        ack_thread = threading.Thread(target=self._process_acks, daemon=True)
        ack_thread.start()

        while not self.shutdown_initiated:
            try:
                self._middleware.basic_consume(
                    self._channel,
                    MLFLOW_QUEUE_NAME,
                    self._on_message,
                    consumer_tag=self._config.pod_name,
                )
                self._middleware.start_consuming(self._channel)
            except (
                pika.exceptions.AMQPConnectionError,
                pika.exceptions.ChannelClosedByBroker,
            ) as e:
                logging.error(f"AMQP Connection error in Listener: {e}")
                if not self.shutdown_initiated:
                    self._reconnect_to_middleware()
            except Exception as e:
                logging.error(f"Error in Listener consumption loop: {e}")
                break

        for worker in self._active_workers:
            worker.join()

    def _reconnect_to_middleware(self):
        self._middleware.connect()
        self._channel = self._middleware.create_channel(
            prefetch_count=self._config.num_workers
        )

    def _process_acks(self):
        """Thread que procesa los ACKs de los workers."""
        while not self.shutdown_initiated:
            try:
                ack_data = self._ack_queue.get(block=True)
                if ack_data is not None:
                    delivery_tag, success = ack_data
                    if success:
                        self._channel.basic_ack(delivery_tag=delivery_tag)
                        logging.info(f"ACK sent for delivery_tag: {delivery_tag}")
                    else:
                        # NACK y requeue=false if the message processing failed
                        self._channel.basic_nack(
                            delivery_tag=delivery_tag, requeue=False
                        )
                        logging.warning(
                            f"NACK sent for delivery_tag: {delivery_tag}, rejecting message"
                        )
                elif ack_data is None:
                    logging.info("ACK thread received shutdown signal.")
                    break
            except Exception:
                logging.exception("Error in ACK processing thread")
                continue

    def start_worker_pool(self):
        for i in range(self._config.num_workers):
            mlflow_logger = MlflowLogger(
                self._workers_queue,
                self._ack_queue,
                self._config.tracking_uri,
                self.tracking_username,
                self.tracking_password,
            )
            mlflow_logger.start()
            logging.info(
                f"Started MlflowLogger worker process with PID {mlflow_logger.pid}"
            )
            self._active_workers.append(mlflow_logger)

    def handle_sigterm(self, signum, frame):
        """Handle SIGTERM signal for graceful shutdown."""
        try:
            logging.info("Received SIGTERM, shutting down gracefully...")
            self.shutdown_initiated = True
            self._middleware.handle_sigterm()
            self._middleware.stop_consuming(self._channel)
            self._ack_queue.put(None) # signal to finish ack_thread
            for _ in self._active_workers:
                self._workers_queue.put((None, None))
            for worker in self._active_workers:
                worker.join()
            self._middleware.close_connection()
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")

    # def _process_input_data(self, data):
    #     target_shape = (28, 28, 1)
    #     target_dtype = np.float32
    #     data_array = np.frombuffer(data, dtype=target_dtype)

    #     data_size = np.prod(target_shape)
    #     num_elements = data_array.size
    #     num_samples = num_elements // data_size

    #     if num_samples * data_size != num_elements:
    #         raise ValueError(
    #             f"Data size incompatible with expected format. "
    #             f"Expected elements per sample: {data_size}, "
    #             f"total elements: {num_elements}, "
    #             f"calculated samples: {num_samples}, "
    #             f"remainder: {num_elements % data_size}"
    #         )

    #     try:
    #         data_array = data_array.reshape((num_samples, *target_shape))
    #     except Exception as e:
    #         raise ValueError(f"Error reshaping data: {e}")

    #     if len(data_array.shape) == 4:
    #         H, W = data_array.shape[1], data_array.shape[2]

    #         if data_array.shape[-1] in [1, 3] and H != 1:
    #             data_array = np.transpose(data_array, (0, 3, 1, 2))

    #     return data_array

    def _on_message(self, ch, method, properties, body):
        try:
            logging.info("Received message for MLflow logging")
            message = mlflow_probs_pb2.MlflowProbs()
            message.ParseFromString(body)
            session_id = message.session_id

            run_id = self._run_registry.get_run_id(session_id)

            if run_id is None:
                os.environ["MLFLOW_TRACKING_USERNAME"] = (
                    self._config.mlflow_tracking_username
                )
                os.environ["MLFLOW_TRACKING_PASSWORD"] = (
                    self._config.mlflow_tracking_password
                )
                mlflow.set_tracking_uri(self._config.tracking_uri)
                mlflow.set_experiment(message.client_id)

                with mlflow.start_run(run_name=session_id) as run:
                    run_id = run.info.run_id

                self._run_registry.save_run_id(session_id, run_id)

            mlflow_data = LoggingDTO(
                client_id=message.client_id,
                session_id=message.session_id,
                run_id=run_id,
                inputs=message.data,
                labels=list(message.labels),
                pred=np.array([list(p.values) for p in message.pred], dtype=np.float32),
                batch_index=message.batch_index,
                delivery_tag=method.delivery_tag,
            )
            self._workers_queue.put(mlflow_data)

        except Exception as e:
            logging.exception("Unhandled exception in _on_message")
            raise e
