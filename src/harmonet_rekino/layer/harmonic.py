import numpy as np
import torch
import torch.nn as nn

from itertools import combinations, combinations_with_replacement

from integrate import hermite_gauss, hermite_gauss_2d

class HarmonicLayer(nn.Module):
    def __init__(self, in_features, out_features, biased=True) -> None:
        super().__init__()

        self.n, self.m = in_features, out_features

        self.w = nn.Parameter(torch.randn(self.n, self.m))
        self.b = 0
        if biased:
            self.b = nn.Parameter(torch.randn(self.m))
    
    @staticmethod
    def act(x):
        return torch.tanh(x)
    
    @staticmethod
    def actprime(x):
        return 1 - torch.tanh(x)**2

    def _qr_decompose(self):
        W = torch.stack([self.w[:, (i, j)] for i, j in combinations(range(self.m), r=2)])
        _, R_triu = torch.linalg.qr(W)
        _, R_diag = torch.linalg.qr(self.w.T[:, :, None])

        return R_triu, torch.squeeze(R_diag)
    
    def _quad(self, w, b, x):
        return self.actprime(w*x + b)**2
    
    def _cube(self, R_triu, b, x, y):
        return self.actprime(R_triu[0, 0]*x + b[0]) * self.actprime(R_triu[0, 1]*x + R_triu[1, 1]*y + b[1])
    
    def compute_tuning(self, deg):
        sigma = torch.zeros(self.m, self.m)
        R_triu, R_diag = self._qr_decompose()

        for i, j in combinations_with_replacement(range(self.m), r=2):
            if i == j:
                sigma[i, i] = np.sqrt(np.pi) * hermite_gauss(lambda x: self._quad(R_diag[i], self.b[i], x), deg)
            else:
                sigma[i, j] = sigma[j, i] = hermite_gauss_2d(lambda x, y: self._cube(R_triu[i], self.b[[i, j]], x, y), deg)
        
        return (self.w.T @ self.w) * sigma


    def forward(self, x):
        return self.act(x @ self.w + self.b)
