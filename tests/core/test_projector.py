import unittest
from pathlib import Path

import torch

from adaptive_ct.core.projector import project_dense_parallel, require_native
from adaptive_ct.data.dataset import load_r2_dataset
from adaptive_ct.evaluation.metrics import projection_metrics


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class ProjectorAngleZeroTests(unittest.TestCase):
    def test_native_projector_matches_r2_angle_zero(self):
        require_native()
        root = Path(r"D:\Projects\3.3DGS\SAD-GS\data\external_baselines\r2_local\figure473_ctgs96_48v")
        ds = load_r2_dataset(root, device="cuda")
        detector_h, detector_w = ds.detector_shape
        pred = project_dense_parallel(
            ds.volume,
            ds.train.angles[:1],
            detector_h=detector_h,
            detector_w=detector_w,
        )[0]
        metrics = projection_metrics(pred, ds.train.projections[0])
        self.assertGreater(metrics["psnr"], 80.0)


if __name__ == "__main__":
    unittest.main()

