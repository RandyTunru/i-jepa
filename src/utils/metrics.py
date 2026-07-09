import time
import torch
import yaml

def get_num_params(model):
    """Returns the total number of trainable parameters in millions."""
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_params / 1e6

class ThroughputMonitor:
    def __init__(self, batch_size, seq_len):
        self.tokens_per_batch = batch_size * seq_len
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def get_tps(self):
        """Calculates Tokens Per Second (TPS)"""
        if self.start_time is None:
            return 0.0
        elapsed = time.time() - self.start_time
        tps = self.tokens_per_batch / elapsed
        self.start_time = time.time() # Reset for next measurement
        return tps