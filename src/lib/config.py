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
        self.port = int(os.getenv("RABBITMQ_PORT", "5672"))
        self.username = os.getenv("RABBITMQ_USER", "guest")
        self.password = os.getenv("RABBITMQ_PASSWORD", "guest")
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.num_workers = int(os.getenv("NUM_WORKERS", "4"))


def initialize_config():   
    return Config() 
