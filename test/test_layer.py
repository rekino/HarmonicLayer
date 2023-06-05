import unittest as ut
import torch
from scipy.integrate import dblquad, quad
import numpy as np

from src.harmonet_rekino.layer import HarmonicLayer


class TestLayer(ut.TestCase):
    def setUp(self) -> None:
        return super().setUp()
    
    def test_qr(self):
        layer = HarmonicLayer(2, 2)
        R_triu, R_diag = layer._qr_decompose()
        R_triu = R_triu.detach().numpy()
        R_diag = R_diag.detach().numpy()
        self.assertEqual(len(R_triu.shape), 3)
        self.assertEqual(R_triu.shape[0], 1)
        self.assertEqual(len(R_diag.shape), 1)
        self.assertEqual(R_diag.shape[0], 2)

        w, b = layer.w.detach().numpy(), layer.b.detach().numpy()

        def integrand_00(x, y):
            return np.tanh(x * w[0, 0] + y * w[1, 0] + b[0])**2 * np.exp(-(x**2 + y**2))
        
        s00, err = dblquad(integrand_00, -np.inf, np.inf, -np.inf, np.inf)

        def integrand_00_rotated(x):
            return np.tanh(x * R_diag[0] + b[0])**2 * np.exp(-x**2)
        
        s00_rotated, err = quad(integrand_00_rotated, -np.inf, np.inf)

        self.assertAlmostEqual(s00, np.sqrt(np.pi) * s00_rotated, 5)

        def integrand_01(x, y):
            return np.tanh(x * w[0, 0] + y * w[1, 0] + b[0]) * np.tanh(x * w[0, 1] + y * w[1, 1] + b[1]) * np.exp(-(x**2 + y**2))
        
        s01, err = dblquad(integrand_01, -np.inf, np.inf, -np.inf, np.inf)

        def integrand_01_rotated(x, y):
            return np.tanh(x * R_triu[0, 0, 0] + b[0]) * np.tanh(x * R_triu[0, 0, 1] + y * R_triu[0, 1, 1] + b[1]) * np.exp(-(x**2 + y**2))
        
        s01_rotated, err = dblquad(integrand_01_rotated, -np.inf, np.inf, -np.inf, np.inf)

        self.assertAlmostEqual(s01, s01_rotated, 5)

    
    def test_tuning(self):
        torch.manual_seed(1)
        layer = HarmonicLayer(2, 2)
        sigma = layer.compute_tuning(20).detach().numpy()

        w, b = layer.w.detach().numpy(), layer.b.detach().numpy()

        def integrand_00(x, y):
            return (1 - np.tanh(x * w[0, 0] + y * w[1, 0] + b[0])**2)**2 * np.exp(-(x**2 + y**2))

        s00, err = dblquad(integrand_00, -np.inf, np.inf, -np.inf, np.inf)
        self.assertAlmostEqual(s00 * (w[0, 0]**2 + w[1, 0]**2), sigma[0, 0], 3)

        def integrand_01(x, y):
            return (1 - np.tanh(x * w[0, 0] + y * w[1, 0] + b[0])**2) * (1 - np.tanh(x * w[0, 1] + y * w[1, 1] + b[1])**2) * np.exp(-(x**2 + y**2))

        s01, err = dblquad(integrand_01, -np.inf, np.inf, -np.inf, np.inf)
        self.assertAlmostEqual(s01 * (w[0, 0]*w[0, 1] + w[1, 0]*w[1, 1]), sigma[0, 1], 3)


    def test_forward(self):
        layer = HarmonicLayer(2, 2)
        out = layer(torch.randn(3, 2)).detach().numpy()

        self.assertEqual(len(out.shape), 2)
        self.assertEqual(out.shape[0], 3)
        self.assertEqual(out.shape[1], 2)