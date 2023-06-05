from numpy.polynomial.hermite import hermgauss
import torch

def hermite_gauss(func, deg):
    x, w = hermgauss(deg)
    x = torch.asarray(x)
    w = torch.asarray(w)
    return func(x) @ w