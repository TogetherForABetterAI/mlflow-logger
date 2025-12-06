import os

MLFLOW_EXCHANGE = "mlflow_exchange"
MLFLOW_QUEUE_NAME = "mlflow_queue"
MLFLOW_ROUTING_KEY = "mlflow.key"
ARTIFACTS_DIR = os.path.join(os.getcwd(), "artifacts")


class Config:
    def __init__(self):
        self.tracking_uri = os.getenv("TRACKING_URI", "http://mlflow:5000")
        self.logging_level = os.getenv("LOGGING_LEVEL", "INFO")
        self.host = os.getenv("RABBITMQ_HOST", "rabbitmq")
        self.pod_name = os.getenv("POD_NAME", "mlflow-logger-pod")
        self.port = int(os.getenv("RABBITMQ_PORT", "5672"))
        self.username = os.getenv("RABBITMQ_USER", "guest")
        self.password = os.getenv("RABBITMQ_PASSWORD", "guest")
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.num_workers = int(os.getenv("NUM_WORKERS", "4"))
        self.postgres_host = os.getenv("MLFLOW_LOGGER_POSTGRES_HOST", "sessions-db")
        self.postgres_port = int(
            os.getenv("MLFLOW_LOGGER_POSTGRES_INTERNAL_PORT", "5432")
        )
        self.mlflow_tracking_username = os.getenv(
            "MLFLOW_TRACKING_USERNAME", "mlflow_user"
        )
        self.mlflow_tracking_password = os.getenv(
            "MLFLOW_TRACKING_PASSWORD", "mlflow_password"
        )

        self.postgres_user = os.getenv("MLFLOW_LOGGER_POSTGRES_USER", "mlflow_user")
        print(f"Postgres User: {self.postgres_user}")
        self.postgres_password = os.getenv(
            "MLFLOW_LOGGER_POSTGRES_PASSWORD", "mlflow_password"
        )
        self.postgres_db = os.getenv("MLFLOW_LOGGER_POSTGRES_DB", "mlflow_db")
        self.db_uri = (
            f"postgresql://{self.postgres_user}:"
            f"{self.postgres_password}@{self.postgres_host}:"
            f"{self.postgres_port}/{self.postgres_db}"
        )


def initialize_config():
    return Config()
