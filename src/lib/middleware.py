import logging
import pika
from lib.config import MLFLOW_EXCHANGE, MLFLOW_QUEUE_NAME, MLFLOW_ROUTING_KEY


class Middleware:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("middleware")
        self._consuming = False
        self.consumer_tag = None
        self._is_running = False
        self._on_callback = True
        try:
            credentials = pika.PlainCredentials(config.username, config.password)
            self.conn = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=config.host,
                    port=config.port,
                    credentials=credentials,
                    heartbeat=5000,
                )
            )
            self.logger.info(
                f"Connected to RabbitMQ at {config.host}:{config.port} as {config.username}"
            )
        except Exception as e:
            self.logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise e

        channel = self.create_channel()

        self.declare_exchange(
            channel, MLFLOW_EXCHANGE, exchange_type="direct", durable=True
        )

        self.declare_queue(channel, MLFLOW_QUEUE_NAME, durable=True)
        self.bind_queue(
            channel, MLFLOW_QUEUE_NAME, MLFLOW_EXCHANGE, routing_key=MLFLOW_ROUTING_KEY
        )

    def is_running(self):
        return self._is_running

    def create_channel(self, prefetch_count=1):
        """Create and return a new channel from the shared connection, with optional prefetch_count."""
        channel = self.conn.channel()
        channel.basic_qos(prefetch_count=prefetch_count)
        return channel

    def declare_queue(self, channel, queue_name: str, durable: bool = False):
        try:
            channel.queue_declare(queue=queue_name, durable=durable)
            self.logger.info(f"Queue '{queue_name}' declared successfully")
        except Exception as e:
            self.logger.error(f"Failed to declare queue '{queue_name}': {e}")
            raise e

    def declare_exchange(
        self,
        channel,
        exchange_name: str,
        exchange_type: str = "direct",
        durable: bool = False,
    ):
        try:
            channel.exchange_declare(
                exchange=exchange_name, exchange_type=exchange_type, durable=durable
            )
            self.logger.info(f"Exchange '{exchange_name}' declared successfully")
        except Exception as e:
            self.logger.error(f"Failed to declare exchange '{exchange_name}': {e}")
            raise e

    def bind_queue(
        self, channel, queue_name: str, exchange_name: str, routing_key: str
    ):
        try:
            channel.queue_bind(
                exchange=exchange_name, queue=queue_name, routing_key=routing_key
            )
            self.logger.info(
                f"Queue '{queue_name}' bound to exchange '{exchange_name}' "
                f"with routing key '{routing_key}'"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to bind queue '{queue_name}' to exchange '{exchange_name}' "
                f"with routing key '{routing_key}': {e}"
            )
            raise e

    def basic_consume(
        self, channel, queue_name: str, callback_function, consumer_tag=None
    ) -> str:
        self.logger.info(f"Setting up consumer for queue: {queue_name}")
        self.consumer_tag = channel.basic_consume(
            queue=queue_name,
            on_message_callback=self.callback_wrapper(callback_function),
            auto_ack=False,
            consumer_tag=consumer_tag,
        )

    def basic_send(
        self,
        channel,
        exchange_name: str,
        routing_key: str,
        body: bytes,
        properties: pika.BasicProperties = None,
    ):
        try:
            channel.basic_publish(
                exchange=exchange_name,
                routing_key=routing_key,
                body=body,
                properties=properties,
            )
        except Exception as e:
            self.logger.error(
                f"Failed to send message to exchange '{exchange_name}' "
                f"with routing key '{routing_key}': {e}"
            )
            raise e

    def on_callback(self):
        return self._on_callback

    def callback_wrapper(self, callback_function):
        self._on_callback = True

        def wrapper(ch, method, properties, body):
            try:
                callback_function(ch, method, properties, body)
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception as e:
                self.logger.error(
                    f"action: rabbitmq_callback | result: fail | error: {e}"
                )
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

            if not self._is_running:
                self.cancel_channel_consuming(ch)

        self._on_callback = False
        return wrapper

    def start_consuming(self, channel):
        try:
            self._is_running = True
            if channel and channel.is_open:
                self.logger.info("Starting to consume messages")
                channel.start_consuming()
        except KeyboardInterrupt:
            self.logger.error("Received interrupt signal, stopping consumption")

    def close_channel(self, channel):
        try:
            if self._is_running:
                raise Exception("Cannot close channel while middleware is running")

            if channel and channel.is_open:
                channel.basic_cancel(self.consumer_tag)
                channel.stop_consuming()
                channel.close()
                self.logger.info("Stopped consuming messages")
        except Exception as e:
            self.logger.error(
                f"action: rabbitmq_channel_close | result: fail | error: {e}"
            )

    def cancel_channel_consuming(self, channel):
        if channel and channel.is_open:
            self.logger.info(f"Cancelling consumer for channel: {channel}")
            channel.basic_cancel(consumer_tag=self.consumer_tag)
            self.channel.stop_consuming()

    def delete_queue(self, channel, queue_name: str):
        try:
            channel.queue_delete(queue=queue_name)
            self.logger.info(f"Queue '{queue_name}' deleted successfully")
        except Exception as e:
            self.logger.error(f"Failed to delete queue '{queue_name}': {e}")

    def stop_consuming(self, consumer_tag: str):
        """
        Stop consuming messages for a specific consumer.
        This allows graceful shutdown by preventing new messages from being consumed.

        Args:
            consumer_tag: The consumer tag to stop)
        """
        self.is_running = False
        try:
            if self.channel and not self.channel.is_closed:
                self.channel.basic_cancel(consumer_tag)
                self.logger.info(f"Consumer {consumer_tag} stopped successfully")
            else:
                self.logger.warning("Channel is already closed, cannot stop consuming")
        except Exception as e:
            self.logger.error(f"Error stopping consumer {consumer_tag}: {e}")
            raise e

    def close_connection(self):
        try:
            if self.connection and not self.connection.is_closed:
                self.connection.close()
                self.logger.info("Connection closed")
        except Exception as e:
            self.logger.error(f"Error closing connection: {e}")
            raise e
