import torch
import torch.nn as nn
import os

class CombinedModel(nn.Module):
    def __init__(self, input_size):
        super(CombinedModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)    
        self.layer2 = nn.Linear(64, 32)            
        self.layer3 = nn.Linear(32, 2)   
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

def combine_models():
    # Load malware model
    malware_checkpoint = torch.load('models/malware_weights/final_model.pt')
    malware_model = CombinedModel(malware_checkpoint['input_size'])
    malware_model.load_state_dict(malware_checkpoint['model_state_dict'])
    malware_permissions = malware_checkpoint['permissions']

    # Load benign model
    benign_checkpoint = torch.load('models/benign_weights/final_model.pt')
    benign_model = CombinedModel(benign_checkpoint['input_size'])
    benign_model.load_state_dict(benign_checkpoint['model_state_dict'])
    benign_permissions = benign_checkpoint['permissions']

    # Combine permissions
    all_permissions = sorted(list(set(malware_permissions + benign_permissions)))
    
    # Create combined model
    combined_model = CombinedModel(len(all_permissions))
    
    # Initialize weights for the combined model
    with torch.no_grad():
        # For each layer in the combined model
        for name, param in combined_model.named_parameters():
            if 'weight' in name:
                # Get corresponding weights from both models
                malware_weight = malware_model.state_dict()[name]
                benign_weight = benign_model.state_dict()[name]
                
                # Create new weight tensor with combined size
                new_weight = torch.zeros_like(param)
                
                # Map weights from malware model
                for i, perm in enumerate(malware_permissions):
                    if perm in all_permissions:
                        j = all_permissions.index(perm)
                        if 'layer1' in name:  # Only map weights for input layer
                            new_weight[:, j] = malware_weight[:, i]
                        else:  # For other layers, use average of both models
                            new_weight = (malware_weight + benign_weight) / 2
                
                # Map weights from benign model
                for i, perm in enumerate(benign_permissions):
                    if perm in all_permissions:
                        j = all_permissions.index(perm)
                        if 'layer1' in name:  # Only map weights for input layer
                            # Average if the permission exists in both models
                            if perm in malware_permissions:
                                new_weight[:, j] = (new_weight[:, j] + benign_weight[:, i]) / 2
                            else:
                                new_weight[:, j] = benign_weight[:, i]
                
                param.data.copy_(new_weight)
            elif 'bias' in name:
                # Average biases
                param.data = (malware_model.state_dict()[name] + benign_model.state_dict()[name]) / 2

    # Save combined model
    if not os.path.exists('models'):
        os.makedirs('models')
    
    torch.save({
        'model_state_dict': combined_model.state_dict(),
        'input_size': len(all_permissions),
        'permissions': all_permissions
    }, 'models/server_model.pt')
    
    print("Combined model saved to models/server_model.pt")
    print(f"Number of permissions in combined model: {len(all_permissions)}")
    print(f"Number of permissions from malware model: {len(malware_permissions)}")
    print(f"Number of permissions from benign model: {len(benign_permissions)}")

if __name__ == "__main__":
    combine_models() 