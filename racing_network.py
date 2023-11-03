import torch
from torch import nn
from torch.nn import Linear, RNN, MultiheadAttention, BatchNorm1d
import numpy as np
from numpy import sqrt
import math
import importlib
import inspect
from pathlib import Path
import torch.nn.functional as F


def get_network_classes():
    filename = Path(__file__).stem
    return [
        (name, cls) for (name, cls) in inspect.getmembers(
            importlib.import_module(filename), inspect.isclass
        )
        if cls.__module__ == filename
    ]


class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.
    
    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
        
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 1.0):
        """Initialization."""
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.
        
        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
    
    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())


class DenseNetwork(nn.Module):
    def __init__(self, param_dict: list, device: str = "cpu"):
        super().__init__()
        
        n_neurons = [
            param_dict["n_inputs"], 
            *param_dict["params"],
            param_dict["n_outputs"]
        ]
        self.n_neurons = n_neurons
        self.layers = nn.ModuleList()
        self.device = device

        for i in range(len(n_neurons)-1):
            self.layers.append(Linear(n_neurons[i], n_neurons[i + 1]))
            
        self.double()
        self.to(device)

    def layer_forward(
        self, 
        x: torch.Tensor, 
        layer: torch.nn.Module
    ) -> tuple:

        edges = (x * layer.weight)
        x = edges.sum(dim=1)[None, :] + layer.bias
        return x, edges

    def forward(self, x: torch.Tensor, return_hidden: bool = False):
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

    def get_min_max(self, x: torch.Tensor, layer_nr: int):
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
    def __init__(self, param_dict: int, device: str = "cpu") -> None:
        super().__init__()

        self.recurrent_layers = nn.ModuleList()
        self.device = device
        n_neurons = [
            param_dict["n_inputs"], 
            *param_dict["params"],
            param_dict["n_outputs"]
        ]

        for i in range(len(n_neurons)-2):
            self.recurrent_layers.append(
                RNN(
                    n_neurons[i], 
                    n_neurons[i + 1], 
                    batch_first=True, 
                    nonlinearity="relu"
                )
            )
        self.linear = Linear(n_neurons[-2], n_neurons[-1])
        self.double()
        self.to(self.device)

    def forward(self, x: torch.Tensor):
        for layer in self.recurrent_layers:
            x, _ = layer(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return x


class AttentionNetwork(nn.Module):
    def __init__(self, param_dict: int, device: str = "cpu"):
        super().__init__()

        self.attention_layers = nn.ModuleList()
        self.device = device
        params = param_dict["params"]
        n_hidden_neurons = params[0]
        n_heads_per_layer = [*params[1:]]

        self.recurrent_layer = RNN(
            param_dict["n_inputs"], 
            n_hidden_neurons, 
            nonlinearity="relu", 
            batch_first=True
        )
        self.recurrent_layer.flatten_parameters()

        for n_heads in n_heads_per_layer:
            layer = MultiheadAttention(
                n_hidden_neurons, 
                n_heads, 
                batch_first=True
            )
            self.attention_layers.append(layer)
        self.linear = Linear(n_hidden_neurons, param_dict["n_outputs"])
        self.global_attn_weights = nn.Parameter(torch.rand(param_dict["seq_length"]))
        self.double()
        self.to(self.device)

    def forward(self, x: torch.Tensor):
        x, _ = self.recurrent_layer(x)

        for self_attn in self.attention_layers:
            x, _ = self_attn(x, x, x)
            x = x.relu()
        
        attention_scores = self.global_attn_weights.softmax(dim=0)
        x = torch.einsum("bij,i->bj", x, attention_scores)
        x = self.linear(x)
        return x


class NoisyDenseNetwork(nn.Module):
    def __init__(self, param_dict: list, device: str = "cpu"):
        super().__init__()
        
        n_neurons = [
            param_dict["n_inputs"], 
            *param_dict["params"],
            param_dict["n_outputs"]
        ]
        self.n_neurons = n_neurons
        self.layers = nn.ModuleList()
        self.device = device

        for i in range(len(n_neurons) - 3):
            self.layers.append(Linear(n_neurons[i], n_neurons[i + 1]))

        for i in range(len(n_neurons) - 3, len(n_neurons) - 1):
            self.layers.append(NoisyLinear(n_neurons[i], n_neurons[i + 1]))
        
        self.double()
        self.to(device)

    def forward(self, x: torch.Tensor):
        x_size = x.size()
        x = x.view((x_size[0], x_size[-1]))

        for layer in self.layers[:-1]:
            x = layer(x)
            x = x.relu()
        x = self.layers[-1](x)
        return x
        
    def reset_noise(self):
        """Reset all noisy layers."""
        self.layers[-2].reset_noise()
        self.layers[-1].reset_noise()


class NoisyAttentionNetwork(nn.Module):
    def __init__(self, param_dict: int, device: str = "cpu"):
        super().__init__()

        self.attention_layers = nn.ModuleList()
        self.device = device
        params = param_dict["params"]
        n_hidden_neurons = params[0]
        n_heads_per_layer = [*params[1:]]

        self.recurrent_layer = RNN(
            param_dict["n_inputs"], 
            n_hidden_neurons, 
            nonlinearity="relu", 
            batch_first=True
        )
        self.recurrent_layer.flatten_parameters()

        for n_heads in n_heads_per_layer:
            layer = MultiheadAttention(
                n_hidden_neurons, 
                n_heads, 
                batch_first=True
            )
            self.attention_layers.append(layer)
        self.noisy_linear1 = NoisyLinear(n_hidden_neurons, n_hidden_neurons)
        self.noisy_linear2 =  NoisyLinear(n_hidden_neurons, param_dict["n_outputs"])
        self.global_attn_weights = nn.Parameter(torch.rand(param_dict["seq_length"]))
        self.double()
        self.to(self.device)

    def forward(self, x: torch.Tensor):
        x, _ = self.recurrent_layer(x)

        for self_attn in self.attention_layers:
            x, _ = self_attn(x, x, x)
            x = x.relu()
        
        attention_scores = self.global_attn_weights.softmax(dim=0)
        x = torch.einsum("bij,i->bj", x, attention_scores)
        x = self.noisy_linear1(x).relu()
        x = self.noisy_linear2(x)
        return x
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.noisy_linear1.reset_noise()
        self.noisy_linear2.reset_noise()

class DenseDuelingNetwork(nn.Module):
    def __init__(self, param_dict: list, device: str = "cpu"):
        super().__init__()
        
        n_neurons = [
            param_dict["n_inputs"], 
            *param_dict["params"],
            param_dict["n_outputs"]
        ]
        self.n_neurons = n_neurons
        self.device = device

        self.feature_layer = Linear(n_neurons[0], n_neurons[1])
        
        self.advantage_layers = nn.ModuleList()
        for i in range(1, len(n_neurons)-3):
            self.advantage_layers.append(Linear(n_neurons[i], n_neurons[i + 1]))
        
        self.advantage_layers.append(NoisyLinear(n_neurons[-3], n_neurons[-2]))
        self.advantage_layers.append(NoisyLinear(n_neurons[-2], n_neurons[-1]))
        
        self.value_layers = nn.ModuleList()
        for i in range(1, len(n_neurons)-3):
            self.value_layers.append(Linear(n_neurons[i], n_neurons[i + 1]))
        
        self.value_layers.append(NoisyLinear(n_neurons[-3], n_neurons[-2]))
        self.value_layers.append(NoisyLinear(n_neurons[-2], 1))
            
        self.double()
        self.to(device)

    def forward(self, x: torch.Tensor):
        x_size = x.size()
        x = x.view((x_size[0], x_size[-1]))

        x = self.feature_layer(x)
        x = x.relu()

        x_adv = x.clone()

        for layer in self.advantage_layers[:-1]:
            x_adv = layer(x_adv)
            x_adv = x_adv.relu()
        advantage = self.advantage_layers[-1](x_adv)

        for layer in self.value_layers[:-1]:
            x = layer(x)
            x = x.relu()
        value = self.value_layers[-1](x)
        
        return value + advantage - advantage.mean(dim=-1, keepdim=True)
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.value_layers[-1].reset_noise()
        self.value_layers[-2].reset_noise()
        self.advantage_layers[-1].reset_noise()
        self.advantage_layers[-2].reset_noise()


class DuelingAttentionNetwork(nn.Module):
    def __init__(self, param_dict: int, device: str = "cpu"):
        super().__init__()

        self.attention_layers = nn.ModuleList()
        self.device = device
        params = param_dict["params"]
        n_hidden_neurons = params[0]
        n_heads_per_layer = [*params[1:]]

        self.recurrent_layer = RNN(
            param_dict["n_inputs"], 
            n_hidden_neurons, 
            nonlinearity="relu", 
            batch_first=True
        )
        self.recurrent_layer.flatten_parameters()

        for n_heads in n_heads_per_layer:
            layer = MultiheadAttention(
                n_hidden_neurons, 
                n_heads, 
                batch_first=True
            )
            self.attention_layers.append(layer)
        self.global_attn_weights = nn.Parameter(torch.rand(param_dict["seq_length"]))

        self.value_noisy_linear1 = NoisyLinear(n_hidden_neurons, n_hidden_neurons)
        self.value_noisy_linear2 =  NoisyLinear(n_hidden_neurons, 1)

        self.advantage_noisy_linear1 = NoisyLinear(n_hidden_neurons, n_hidden_neurons)
        self.advantage_noisy_linear2 =  NoisyLinear(n_hidden_neurons, param_dict["n_outputs"])
        
        self.double()
        self.to(self.device)

    def forward(self, x: torch.Tensor):
        x, _ = self.recurrent_layer(x)

        for self_attn in self.attention_layers:
            x, _ = self_attn(x, x, x)
            x = x.relu()
        
        attention_scores = self.global_attn_weights.softmax(dim=0)
        x = torch.einsum("bij,i->bj", x, attention_scores)
        x_adv = x.clone()
        
        x = self.value_noisy_linear1(x).relu()
        value = self.value_noisy_linear2(x)

        x_adv = self.advantage_noisy_linear1(x_adv).relu()
        advantage = self.advantage_noisy_linear2(x_adv)

        return value + advantage - advantage.mean(dim=-1, keepdim=True)
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.value_noisy_linear1.reset_noise()
        self.value_noisy_linear2.reset_noise()   
        self.advantage_noisy_linear1.reset_noise()
        self.advantage_noisy_linear2.reset_noise()