import torch
from torch import nn
from torch.nn import Linear, RNN, MultiheadAttention, BatchNorm1d
import numpy as np
from numpy import sqrt
import importlib
import inspect
from pathlib import Path


def get_classes():
    filename = Path(__file__).stem
    return [
        (name, cls) for (name, cls) in inspect.getmembers(
            importlib.import_module(filename), inspect.isclass
        )
        if cls.__module__ == filename
    ]


class DenseNetwork(nn.Module):
    def __init__(self, n_neurons: list, device: str = "cpu"):
        super().__init__()
        
        self.n_neurons = n_neurons
        self.layers = nn.ModuleList()
        self.device = device

        for i in range(len(n_neurons)-1):
            self.layers.append(Linear(n_neurons[i], n_neurons[i + 1]))
        self.double()
        self.to(device)

    def layer_forward(
        self, 
        x: torch.tensor, 
        layer: torch.nn.Module
    ) -> tuple:

        edges = (x * layer.weight)
        x = edges.sum(dim=1)[None, :] + layer.bias
        return x, edges

    def forward(self, x: torch.tensor, return_hidden: bool = False):
        x_size = x.size()
        x = x.view((x_size[0], x_size[-1]))

        if not return_hidden:
            for layer in self.layers[:-1]:
                x = layer(x)
                x = x.relu()
            x = self.layers[-1](x)
            return x
        
        else:
            edges = [None for _ in range(len(self.layers))]
            h_states = [None for _ in range(len(self.layers) + 1)]
            h_states[0] = x.clone()
            for i, layer in enumerate(self.layers[:-1]):
                x, edges[i] = self.layer_forward(x, layer)
                x = x.relu()
                h_states[i + 1] = x.clone()
            x, edges[-1] = self.layer_forward(x, self.layers[-1])
            h_states[-1] = x.clone()
            return x, h_states, edges
    
    def prune_dead_neurons(self, neuron_means: list, neuron_stds: list = None, tol: float = 0.01):
        n_neurons_prev = self.n_neurons[0]
        where_active_prev = torch.ones(n_neurons_prev, dtype=torch.bool)

        for i in range(len(neuron_means)):
            where_active = neuron_means[i] != 0
            if neuron_stds is not None:
                where_active = np.logical_and(where_active, neuron_stds[i] > tol)
            n_neurons = int(where_active.sum())
            new_linear = Linear(n_neurons_prev, n_neurons)
            where_active = torch.from_numpy(where_active)
            weights = self.layers[i].weight[where_active].clone()
            weights = weights[:, where_active_prev]
            with torch.no_grad():
                new_linear.weight = nn.Parameter(weights)
                new_linear.bias = nn.Parameter(self.layers[i].bias[where_active].clone())
                self.layers[i] = new_linear
            n_neurons_prev = n_neurons
            where_active_prev = where_active.clone()
            self.n_neurons[i + 1] = n_neurons
        
        new_linear = Linear(n_neurons_prev, self.n_neurons[-1])
        new_linear.weight = nn.Parameter(self.layers[-1].weight[:, where_active_prev].clone())
        new_linear.bias = nn.Parameter(self.layers[-1].bias.clone())
        self.layers[-1] = new_linear

    def get_min_max(self, x: torch.tensor, layer_nr: int):
        layer = self.layers[layer_nr]
        weights = layer.weight.clone().detach()
        biases = layer.bias[None, :]

        weights_pos = weights.clone().T
        weights_neg = weights.clone().T
        weights_pos[weights_pos <= 0] = 0
        weights_neg[weights_neg > 0] = 0

        x_pos = x.clone()
        x_neg = x.clone()
        x_pos[x_pos <= 0] = 0
        x_neg[x_neg > 0] = 0

        x_max = torch.matmul(x_pos, weights_pos) + torch.matmul(x_neg, weights_neg) + biases
        x_min = torch.matmul(x_pos, weights_neg) + torch.matmul(x_neg, weights_pos) + biases

        return x_min.min(dim=0, keepdim=True)[0].relu(), x_max.max(dim=0, keepdim=True)[0].relu()

    def prune_verified_dead_neurons(self):
        x_min = torch.zeros((1, 7), dtype=torch.double).to(self.device)
        x_max = torch.ones((1, 7), dtype=torch.double).to(self.device)
        """x_min[0, :4] *= 0.1
        x_max[0, :4] *= 0.1
        x_min[0, 4] *= 0.5
        x_max[0, 4] *= 0.5"""
        x_min[0, -2] = -1
        n_used_layers = len(self.layers[:-1])
        active_neurons = [None for _ in range(n_used_layers)]

        for layer_nr in range(n_used_layers):
            x = torch.cat((x_min, x_max), dim=0)
            x_min, x_max = self.get_min_max(x, layer_nr)
            active_neurons[layer_nr] = x_max[0] > 0
        
        n_neurons_prev = self.n_neurons[0]
        where_active_prev = torch.ones(n_neurons_prev, dtype=torch.bool)

        for i in range(n_used_layers):
            where_active = active_neurons[i]
            n_neurons = int(where_active.sum())
            new_linear = Linear(n_neurons_prev, n_neurons)
            weights = self.layers[i].weight[where_active].clone()
            weights = weights[:, where_active_prev]
            with torch.no_grad():
                new_linear.weight = nn.Parameter(weights)
                new_linear.bias = nn.Parameter(self.layers[i].bias[where_active].clone())
                self.layers[i] = new_linear
            n_neurons_prev = n_neurons
            where_active_prev = where_active.clone()
            self.n_neurons[i + 1] = n_neurons
        
        new_linear = Linear(n_neurons_prev, self.n_neurons[-1])
        new_linear.weight = nn.Parameter(self.layers[-1].weight[:, where_active_prev].clone())
        new_linear.bias = nn.Parameter(self.layers[-1].bias.clone())
        self.layers[-1] = new_linear


class RecurrentNetwork(nn.Module):
    def __init__(self, n_neurons: int, device: str = "cpu") -> None:
        super().__init__()

        self.recurrent_layers = nn.ModuleList()
        self.device = device

        for i in range(len(n_neurons)-2):
            self.recurrent_layers.append(RNN(n_neurons[i], n_neurons[i + 1], batch_first=True, nonlinearity="relu"))
        self.linear = Linear(n_neurons[-2], n_neurons[-1])
        self.double()
        self.to(self.device)

    def forward(self, x: torch.tensor):
        for layer in self.recurrent_layers:
            x, _ = layer(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return x


class DenseLookAheadNetwork(nn.Module):
    def __init__(self, n_neurons: int, device: str = "cpu"):
        super().__init__()

        self.layers = nn.ModuleList()
        self.device = device

        for i in range(len(n_neurons)-1):
            self.layers.append(Linear(n_neurons[i], n_neurons[i + 1]))
        self.double()
        self.to(device)

    def forward(self, x: torch.tensor):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = x.relu()
        x = self.layers[-1](x)
        return x


class AttentionNetwork(nn.Module):
    def __init__(self, n_neurons: int, device: str = "cpu"):
        super().__init__()

        self.attention_layers = nn.ModuleList()
        self.device = device
        self.recurrent_layer = RNN(n_neurons[0], n_neurons[1], nonlinearity="relu")
        n_features = n_neurons[1]

        for i in range(2, len(n_neurons)-1):
            layer = MultiheadAttention(n_features, n_neurons[i])
            self.attention_layers.append(layer)
        self.linear = Linear(n_features, n_neurons[-1])
        self.double()
        self.to(self.device)

    def forward(self, x: torch.Tensor):
        x, _ = self.recurrent_layer(x)

        for self_attn in self.attention_layers:
            x, _ = self_attn(x, x, x)
            x = x.relu()
            
        x = x.mean(dim=1)
        x = self.linear(x)
        return x
