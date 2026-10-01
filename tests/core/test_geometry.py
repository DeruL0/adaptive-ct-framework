import unittest
from math import log10
from pathlib import Path

import numpy as np


class GeometryConventionTests(unittest.TestCase):
    def test_r2_angle_zero_is_x_axis_sum_transposed(self):
        root = Path(r"D:\Projects\3.3DGS\SAD-GS\data\external_baselines\r2_local\figure473_ctgs96_48v")
        vol = np.load(root / "vol_gt.npy").astype(np.float32)
        proj = np.load(root / "proj_train" / "proj_train_0000.npy").astype(np.float32)
        pred = vol.sum(axis=0).T * (2.0 / vol.shape[0])
        mse = float(np.mean((pred - proj) ** 2))
        psnr = 20.0 * log10(float(proj.max())) - 10.0 * log10(max(mse, 1e-20))
        self.assertGreater(psnr, 100.0)


if __name__ == "__main__":
    unittest.main()

