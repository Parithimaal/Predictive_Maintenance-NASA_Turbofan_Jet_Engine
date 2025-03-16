import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# A boilerplate class for creating a dataset in PyTorch
class RegressionDataset(Dataset): 
    def __init__(self, X, y):
        self.X = X  # Features 
        self.y = y  # Targets 

    def __len__(self):
        return len(self.X)  # Returns the total number of samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]  # Retrieves one sample at index idx

    
# Defining model
class DynamicSeqRegNN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], activation=nn.ReLU(), dropout=False):
        super(DynamicSeqRegNN, self).__init__()
        self.output_size=1
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout

        layers =[]
        prev_size = self.input_size
        for h_num, h_size in enumerate(self.hidden_sizes):
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(activation)
            if dropout:
                layers.append(nn.Dropout(self.dropout))
            prev_size = h_size


        layers.append(nn.Linear(prev_size, self.output_size))
        self.model = nn.Sequential(*layers)

        self._initialize_weights()

    def forward(self, x):
        return self.model(x) 
    
    def _initialize_weights(self):
        """
        Initialize weights for all layers.
        Uses Kaiming Initialization for ReLU-based networks.
        """
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')  # He initialization
                nn.init.zeros_(layer.bias)  # Bias initialized to zero