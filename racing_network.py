import torch
from torch import nn
from torch.nn import Linear, RNN
from numpy import sqrt
from init_cuda import init_cuda


device = init_cuda()


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


class RecurrentNetwork(nn.Module):
    def __init__(self, n_neurons) -> None:
        super().__init__()

        self.recurrent_layers = nn.ModuleList()

        for i in range(len(n_neurons)-2):
            self.recurrent_layers.append(RNN(n_neurons[i], n_neurons[i+1], batch_first=True, nonlinearity="relu"))
        self.linear = Linear(n_neurons[-2], n_neurons[-1])
        self.double()
        self.to(device)

    def forward(self, x):
        for layer in self.recurrent_layers:
            x, _ = layer(x)
        x = x[:, -1, :]
        x = self.linear(x)
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
