import flwr as fl
import numpy as np
from flwr.server.strategy import FedAvg
import torch
import os
from datetime import datetime
import torch.nn as nn
import signal
import sys
import logging
from typing import List, Tuple, Optional
import socket
from contextlib import closing

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def signal_handler(sig, frame):
    print('Stopping server...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

class MalwareModel(nn.Module):
    def __init__(self, input_size, num_classes=2):
        super(MalwareModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)    
        self.layer2 = nn.Linear(64, 32)            
        self.layer3 = nn.Linear(32, num_classes)   
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(64)      
        self.batch_norm2 = nn.BatchNorm1d(32)      

    def forward(self, x):
        x = self.layer1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer3(x)
        x = self.softmax(x)
        return x

def find_free_port(start_port: int = 8080, max_attempts: int = 10) -> int:
    """Find a free port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            try:
                sock.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"Could not find a free port after {max_attempts} attempts")

def main():
    try:
        # Define strategy
        strategy = FedAvg(
            min_available_clients=1,  # Minimum number of clients to start training
            min_fit_clients=1,  # Minimum number of clients to train
            min_evaluate_clients=1,  # Minimum number of clients to evaluate
            on_fit_config_fn=lambda _: {"epochs": 5},  # 5 epochs per round
            on_evaluate_config_fn=lambda _: {"epochs": 1},  # 1 epoch for evaluation
            initial_parameters=None,  # No initial parameters
            fit_metrics_aggregation_fn=None,  # Don't aggregate metrics
            evaluate_metrics_aggregation_fn=None,  # Don't aggregate metrics
        )

        # Use fixed port 8080
        port = 8080
        logging.info(f"Starting server on port {port}...")
        logging.info("Server will run for 3 rounds")
        logging.info("Waiting for clients to connect...")

        # Start server
        fl.server.start_server(
            server_address=f"127.0.0.1:{port}",
            config=fl.server.ServerConfig(num_rounds=6),
            strategy=strategy
        )

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 