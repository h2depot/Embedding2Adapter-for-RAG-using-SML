from collections import namedtuple

import torch.nn as nn


SamplerOutput = namedtuple("SamplerOutput", ["weight", "bias"])


def init_linear_layer(linear_layer, std=1e-2):
    """Initializes the given linear module as explained in adapter paper."""
    nn.init.normal_(linear_layer.weight, std=std)
    nn.init.zeros_(linear_layer.bias)


def linear_layer(input_dim, output_dim, std=1e-2):
    """Generates a linear module and initializes it."""
    linear = nn.Linear(input_dim, output_dim)
    init_linear_layer(linear, std=std)
    return linear


class AdapterLayersHyperNet(nn.Module):
    def __init__(self, embd_dim: int, input_dim: int, output_dim: int, std=1e-2):
        super().__init__()
        self.embd_dim = embd_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_generator = linear_layer(
            self.embd_dim, self.input_dim * self.output_dim, std=std
        )
        self.bias_generator = linear_layer(
            self.embd_dim, self.output_dim, std=std
        )

    def forward(self, embd):
        weight = self.weight_generator(embd).view(
            self.input_dim, self.output_dim
        )
        bias = self.bias_generator(embd).view(-1)
        return SamplerOutput(weight=weight, bias=bias)


class TaskHyperNet(nn.Module):
    def __init__(self, task_hidden_dim: int, projected_embd_dim: int, embd_dim: int):
        super().__init__()
        self.task_hidden_dim = task_hidden_dim
        self.projected_embd_dim = projected_embd_dim
        self.task_embedding_generator = nn.Sequential(
            linear_layer(embd_dim, self.task_hidden_dim),
            nn.ReLU(),
            linear_layer(self.task_hidden_dim, self.projected_embd_dim),
        )

    def forward(self, task_embedding):
        task_embedding = task_embedding.view(-1)
        return self.task_embedding_generator(task_embedding).view(-1)


class LayerNormHyperNet(nn.Module):
    def __init__(self, projected_embd_dim: int, input_dim: int):
        super().__init__()
        self.projected_embd_dim = projected_embd_dim
        self.weight_generator = linear_layer(self.projected_embd_dim, input_dim)
        self.bias_generator = linear_layer(self.projected_embd_dim, input_dim)

    def forward(self, input):
        return self.weight_generator(input), self.bias_generator(input)
