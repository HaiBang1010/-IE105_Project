import flwr as fl
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
import logging
import socket
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

class BenignModel(nn.Module):
    def __init__(self, input_size, num_classes=2):
        super(BenignModel, self).__init__()
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

def load_data():
    # Load only benign dataset
    df = pd.read_csv('data/datacsv/benign_permissions.csv', low_memory=False)
    
    # Get permissions from benign dataset
    all_permissions = set()
    for perms in df['permissions']:
        all_permissions.update(eval(perms))
    
    # Sort permissions to ensure consistent order
    all_permissions = sorted(list(all_permissions))
    
    # Create binary feature matrix
    X = np.zeros((len(df), len(all_permissions)))
    for i, perms in enumerate(df['permissions']):
        perms_list = eval(perms)
        for j, perm in enumerate(all_permissions):
            if perm in perms_list:
                X[i, j] = 1
    
    # Create labels (all benign = 0)
    y = np.zeros(len(df))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Print data information
    print("\nBenign Data Information:")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    print("Number of unique permissions:", len(all_permissions))
    
    return X_train, X_test, y_train, y_test, all_permissions

class BenignClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test, client_id, all_permissions):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.client_id = client_id
        self.round = 0
        self.all_permissions = all_permissions
        
        # Create directory for saving weights
        self.weights_dir = f"models/benign_weights"
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)

    def get_parameters(self, config):
        # Send model weights to server
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        # Skip setting parameters from server
        # This ensures we keep our own weights
        pass

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Convert data to PyTorch tensors
        X_train_tensor = torch.FloatTensor(self.X_train).to(self.device)
        y_train_tensor = torch.LongTensor(self.y_train).to(self.device)
        
        # Create DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Training loop
        self.model.train()
        total_loss = 0
        for epoch in range(5):  # 5 epochs per round
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            total_loss += epoch_loss
            print(f"Client {self.client_id} - Round {self.round} - Epoch {epoch+1}/5 - Loss: {epoch_loss/len(train_loader):.4f}")

        # Save model weights after training (only in the last round)
        if self.round == 2:  # Last round (0-based indexing)
            weights_path = os.path.join(self.weights_dir, "final_model.pt")
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'input_size': self.X_train.shape[1],
                'permissions': self.all_permissions
            }, weights_path)
            print(f"Saved final model weights to {weights_path}")
            
        self.round += 1
        return self.get_parameters({}), len(self.X_train), {"loss": total_loss/5}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        criterion = nn.CrossEntropyLoss()

        # Convert test data to PyTorch tensors
        X_test_tensor = torch.FloatTensor(self.X_test).to(self.device)
        y_test_tensor = torch.LongTensor(self.y_test).to(self.device)

        # Evaluation
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            loss = criterion(outputs, y_test_tensor)
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == y_test_tensor).float().mean()
            
            # Calculate additional metrics
            true_positives = ((predictions == 1) & (y_test_tensor == 1)).sum().item()
            true_negatives = ((predictions == 0) & (y_test_tensor == 0)).sum().item()
            false_positives = ((predictions == 1) & (y_test_tensor == 0)).sum().item()
            false_negatives = ((predictions == 0) & (y_test_tensor == 1)).sum().item()
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\nClient {self.client_id} - Round {self.round} Evaluation:")
        print(f"Loss: {loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}\n")

        # Save model weights after final evaluation
        if self.round == 2:  # Last round (0-based indexing)
            weights_path = os.path.join(self.weights_dir, "final_model.pt")
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'input_size': self.X_train.shape[1],
                'permissions': self.all_permissions
            }, weights_path)
            print(f"Saved final model weights to {weights_path}")

        # Exit after completing evaluate round 3
        if self.round == 3:
            print(f"Client {self.client_id} completed all rounds. Exiting...")
            sys.exit(0)

        return float(loss), len(self.X_test), {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score)
        }

def connect_to_server(server_address: str, max_retries: int = 5, retry_delay: int = 2) -> bool:
    """Try to connect to the server with retries."""
    for attempt in range(max_retries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect(('127.0.0.1', int(server_address.split(':')[1])))
            return True
        except (socket.error, ValueError) as e:
            if attempt < max_retries - 1:
                logging.info(f"Connection attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logging.error(f"Failed to connect to server after {max_retries} attempts: {str(e)}")
                return False
    return False

def main():
    try:
        # Get client ID from command line argument
        if len(sys.argv) != 2:
            logging.error("Please provide client ID as command line argument")
            sys.exit(1)

        client_id = int(sys.argv[1])
        logging.info(f"Starting benign client {client_id}...")

        # Load data for this client
        X_train, X_test, y_train, y_test, all_permissions = load_data()

        # Create model with correct input size
        input_size = len(all_permissions)  # Use actual number of permissions
        model = BenignModel(input_size)
        
        # Initialize model weights
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        model.apply(init_weights)

        # Create client
        client = BenignClient(model, X_train, y_train, X_test, y_test, client_id, all_permissions)

        # Try to connect to server
        server_address = "127.0.0.1:8080"
        max_retries = 5
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                # Start client
                fl.client.start_numpy_client(
                    server_address=server_address,
                    client=client
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logging.info(f"Connection attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logging.error(f"Failed to connect to server after {max_retries} attempts: {str(e)}")
                    sys.exit(1)

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 