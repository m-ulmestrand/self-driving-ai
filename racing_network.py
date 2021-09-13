import torch
from torch import nn
from torch.nn import Linear
from numpy import sqrt


if torch.cuda.is_available():
    print("CUDA available. Neural calculations run on GPU.")
    device = torch.device("cuda:0")
else:
    print("CUDA unavailable. Neural calculations run on CPU.")
    device = torch.device("cpu")


class DenseNetwork(nn.Module):
    def __init__(self, n_neurons):
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(len(n_neurons)-1):
            self.layers.append(Linear(n_neurons[i], n_neurons[i+1]))
        self.double()
        self.to(device)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = x.relu()
        x = self.layers[-1](x)
        return x


class DenseLookAheadNetwork(nn.Module):
    def __init__(self, n_neurons):
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(len(n_neurons)-1):
            self.layers.append(Linear(n_neurons[i], n_neurons[i+1]))
        self.double()
        self.to(device)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = x.relu()
        x = self.layers[-1](x)
        return x
