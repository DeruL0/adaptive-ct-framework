import unittest

import torch

from adaptive_ct.latent_p1 import LatentP1Octree


def _sample_parent_points() -> torch.Tensor:
    local = torch.linspace(0.05, 0.95, 5)
    return torch.stack(torch.meshgrid(local, local, local, indexing="ij"), dim=-1).reshape(-1, 3) * 2.0 - 1.0


class LatentP1Tests(unittest.TestCase):
    def test_split_initialization_improves_on_repeating_parent_latent(self):
        torch.manual_seed(7)
        model = LatentP1Octree(levels=[1, 2], latent_dim=4, attenuation_shift=-3.0)
        with torch.no_grad():
            model.leaf_latents.normal_()
            model.decoder.weight.normal_()
            model.decoder.bias.normal_()
        points = _sample_parent_points()
        before = model.forward_mu(points)
        parent_latent = model.leaf_latents[0].detach().clone()

        model.split_leaf(0)
        fitted_error = torch.mean((model.forward_mu(points) - before).square())

        with torch.no_grad():
            model.leaf_latents[-8:] = parent_latent
        repeated_error = torch.mean((model.forward_mu(points) - before).square())

        self.assertLess(float(fitted_error.detach()), float(repeated_error.detach()))


if __name__ == "__main__":
    unittest.main()
