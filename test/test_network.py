import unittest as ut
import torch

from scipy.integrate import dblquad

from network import MLHP

class TestNetwork(ut.TestCase):

    def test_forward(self):
        model = MLHP(4, 3, 2)
        x = torch.randn(5, 4)
        out = model(x).detach().numpy()

        self.assertEqual(len(out.shape), 2)
        self.assertEqual(out.shape[0], 5)
        self.assertEqual(out.shape[1], 2)
    
    def test_energy(self):
        model = MLHP(2, 2, 1)

        e_hat = model.energy().detach().numpy()

        def integrand(x, y):
            z = torch.asarray([x, y], requires_grad=True)
            out = model(z).sum()
            g = torch.autograd.grad(out, z)[0]

            norm = (g**2).sum().detach().numpy()
            weight = torch.exp(-(z**2).sum()).detach().numpy()
            return norm * weight
        
        e, err = dblquad(integrand, -torch.inf, torch.inf, -torch.inf, torch.inf)

        self.assertAlmostEqual(e, e_hat, 1)
