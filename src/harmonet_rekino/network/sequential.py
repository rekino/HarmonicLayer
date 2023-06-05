import torch
import torch.nn as nn

from harmonet_rekino.layer import HarmonicLayer

class HarmoNet(nn.Module):
    def __init__(self, *dims) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        for n, m in zip(dims[:-2], dims[1:-1]):
            self.layers.append(HarmonicLayer(n, m))
        self.linear = nn.Linear(dims[-2], dims[-1])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.linear(x)

    def energy(self):
        e = 0
        vec = self.linear.weight.T
        for layer in reversed(self.layers):
            sigma = layer.compute_tuning(deg=20)
            e += torch.trace(vec.T @ sigma @ vec)
            vec = layer.w
        return e
