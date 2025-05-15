# Federated Learning with Flower Framework

This project implements federated learning using the Flower Framework to train a malware detection model on the Drebin dataset.

## Setup

0. Run env
```bash
source venv/Scripts/activate
```

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Project

1. Start the server in one terminal:
```bash
python server.py
```

2. Start multiple clients in different terminals (at least 2 clients are required):
```bash
python client.py
```

You can start multiple clients by running the client script in different terminals. Each client will automatically get a different portion of the dataset.

## Project Structure

- `server.py`: Contains the Flower server implementation
- `client.py`: Contains the client implementation with the neural network model
- `requirements.txt`: Lists all required Python packages
- `drebin-215-dataset-5560malware-9476-benign.csv`: The dataset file

## Model Architecture

The project uses a simple neural network with the following architecture:
- Input layer: Size depends on the number of features
- Hidden layer 1: 128 neurons with ReLU activation
- Hidden layer 2: 64 neurons with ReLU activation
- Output layer: 1 neuron with sigmoid activation

## Training Process

The federated learning process works as follows:
1. The server initializes the global model
2. Each client trains the model on their local data
3. The server aggregates the model updates from all clients
4. The process repeats for the specified number of rounds

The training includes:
- 5 epochs per round for each client
- 3 federated learning rounds
- Batch size of 32
- Adam optimizer
- Binary Cross Entropy loss function 