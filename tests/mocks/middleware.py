
from unittest.mock import Mock
from src.proto.mlflow_probs_pb2 import MlflowProbs, PredictionList
class FakeMiddleware:
    def __init__(self, config):
        self._is_running = False
        self.callback = None

    def is_running(self):
        return self._is_running

    def create_channel(self, prefetch_count=1):
        pass

    def declare_queue(self, channel, queue_name: str, durable: bool = False):
        pass
   
    def declare_exchange(self, channel, exchange_name: str, exchange_type: str = "direct", durable: bool = False):
        pass

    def bind_queue(
        self, channel, queue_name: str, exchange_name: str, routing_key: str
    ):
        pass

    def basic_consume(self, channel, queue_name: str, callback_function) -> str:
        self.callback = callback_function
    
    def basic_send(
        self, 
        channel,
        exchange_name: str,
        routing_key: str,
        body: bytes,
    ):
        pass 
    
    def on_callback(self):
        pass
        
    def callback_wrapper(self, callback_function):
        pass

    def start_consuming(self, channel):
        self._is_running = True
        # Generate fake MlflowProbs message
        for i in range(10):
            message = MlflowProbs()
            message.client_id = "acde070d-8c4c-4f0d-9d8a-162843c10333"
            message.batch_index = i
            pred = PredictionList()
            pred.values.extend([0.1, 0.8, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            message.pred.append(pred)
            self.callback(Mock(), Mock(), Mock(), message.SerializeToString())


    def close_channel(self, channel):
        pass

    def cancel_channel_consuming(self, channel):
        pass
            
    def stop_consuming(self):
        pass

    def delete_queue(self, channel, queue_name: str):
        pass

    def close_connection(self):
        pass