import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class DefaultHead(nn.Module):

    def __init__(self, d_model, n_classes, mode):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.linear = nn.Linear(d_model, n_classes)

    def forward(self, x):
        return self.linear(x)
