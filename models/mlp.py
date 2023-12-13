""" src: chatgpt """

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.2):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()  # List to hold the layers

        # First layer that takes the input
        self.layers.append(nn.Linear(input_dim, hidden_dim))

        # Add 'num_layers-2' hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Final layer that outputs the class predictions
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers.append(nn.Dropout(dropout))

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x):
        for layer in self.layers[:-1]:  # Go through all layers except the last one
            x = F.relu(layer(x))  # Apply ReLU activation after each layer

        # No activation after the last layer
        x = self.layers[-1](x)
        return x

if __name__=="__main__":
	# Example usage
	input_dim = 784  # for MNIST dataset
	hidden_dim = 500  # number of nodes in the hidden layer
	output_dim = 10  # number of output classes
	num_layers = 3  # total number of layers in the network

	mlp = MLP(input_dim, hidden_dim, output_dim, num_layers)

