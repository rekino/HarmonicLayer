import unittest as ut
from scipy.integrate import dblquad, quad
import numpy as np

from integrate import hermite_gauss, hermite_gauss_2d


class TestIntegrator(ut.TestCase):
    def setUp(self) -> None:
        return super().setUp()
    
    def test_integrate_1D(self):
        func = lambda x: np.tanh(x + 1)
        approximated = hermite_gauss(func, 20)
        func = lambda x: np.tanh(x + 1) * np.exp(-x**2)
        precise, err = quad(func, -np.inf, np.inf)

        self.assertAlmostEqual(approximated.detach().numpy(), precise, 5)
    
    def test_integrate_2D(self):
        func = lambda x, y: np.tanh(x + y + 1)
        approximated = hermite_gauss_2d(func, 20)
        func = lambda x, y: np.tanh(x + y + 1) * np.exp(-(x**2 + y**2))
        precise, err = dblquad(func, -np.inf, np.inf, -np.inf, np.inf)

        self.assertAlmostEqual(approximated.detach().numpy(), precise, 5)