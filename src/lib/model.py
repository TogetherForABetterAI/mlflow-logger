class LoggingDTO:
    def __init__(self, client_id: str, session_id: str, run_id: str, inputs: list, labels: list, pred: list, batch_index: int):
        self.client_id = client_id
        self.session_id = session_id
        self.run_id = run_id
        self.inputs = inputs
        self.labels = labels
        self.pred = pred
        self.batch_index = batch_index
        

        