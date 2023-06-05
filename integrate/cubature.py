from numpy.polynomial.hermite import hermgauss
import torch

def hermite_gauss_2d(func, deg):
    x, w = hermgauss(deg)
    x = torch.asarray(x)
    w = torch.asarray(w)
    X, Y = torch.meshgrid(x, x)
    W = w[:, None] @ w[None, :]

    return (W * func(X, Y)).sum()
