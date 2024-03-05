import sys
import unittest

import numpy as np

sys.path.append('./')

from source.features import pure_poli_features, gaussian_kernel_features

class TestFeatures(unittest.TestCase):
    def test_pure_poli_features(self):
        # prepare the data:
        expected = np.array(
            [
                [1, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 2, 4, 8],
            ]
        )
        actual = pure_poli_features(np.arange(3), 4)
        self.assertTrue(np.allclose(actual, expected))
    
    def test_gaussian_kernel_features(self):
        # prepare the data:
        expected = np.array(
            [
                [ 0.18846226,  0.05777485],
                [-0.77123494, -0.10430503],
                [-0.52870649,  0.13053439],
            ]
        )
        actual = gaussian_kernel_features(
            np.array([np.pi, 2*np.pi, 3*np.pi]), 
            order=2,
            lscale=1,
            domain_bound=1,
        )
        self.assertTrue(np.allclose(actual, expected))
