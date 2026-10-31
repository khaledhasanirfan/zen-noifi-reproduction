"""Scientific invariants and false-success regression checks."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_noifi import Classifier, attack, metrics, score_noifi


class SumFeatures(nn.Module):
    def feature(self, x):
        # Permutation-invariant representation must have zero MED.
        return x.flatten(1).sum(1, keepdim=True)


class ZeroDenoiser(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


class ExperimentsTest(unittest.TestCase):
    def test_permutation_invariant_representation_has_zero_med(self):
        x = torch.arange(3*16, dtype=torch.float32).reshape(3,1,4,4)
        scores, _ = score_noifi(SumFeatures(), ZeroDenoiser(), x, 5, 42)
        np.testing.assert_allclose(scores, 0, atol=1e-5)

    def test_attack_constraints_and_no_mutation(self):
        torch.manual_seed(42)
        net = Classifier().eval()
        x = torch.rand(4,1,64,64)
        before = x.clone()
        y = torch.tensor([0,1,2,3])
        for method in ['FGSM', 'PGD']:
            adv = attack(net, x, y, method, 0.1, torch.Generator().manual_seed(7))
            self.assertTrue(torch.equal(x, before))
            self.assertLessEqual(float((adv-x).abs().max()), 0.100001)
            self.assertGreaterEqual(float(adv.min()), 0)
            self.assertLessEqual(float(adv.max()), 1)

    def test_threshold_depends_only_on_calibration(self):
        calibration = np.arange(20.)
        a = metrics(np.array([1.,2.]), np.array([25.,26.]), calibration, 0.05)
        b = metrics(np.array([100.,200.]), np.array([250.,260.]), calibration, 0.05)
        self.assertEqual(a['threshold'], b['threshold'])
        self.assertEqual(a['observed_test_fpr'], 0)
        self.assertEqual(b['observed_test_fpr'], 1)

    def test_checker_rejects_equal_nonfinite_arrays(self):
        script = Path(__file__).resolve().parents[1]/'scripts/compare_outputs.py'
        with tempfile.TemporaryDirectory() as d:
            ref, rec = Path(d)/'ref.npy', Path(d)/'rec.npy'
            for values in [np.array([np.inf]), np.array([np.nan]), np.array([])]:
                np.save(ref, values); np.save(rec, values)
                result = subprocess.run([sys.executable, str(script), str(ref), str(rec)], capture_output=True)
                self.assertNotEqual(result.returncode, 0)
            np.save(ref, np.array([1.,2.])); np.save(rec, np.array([1.,2.]))
            result = subprocess.run([sys.executable, str(script), str(ref), str(rec)], capture_output=True)
            self.assertEqual(result.returncode, 0)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
