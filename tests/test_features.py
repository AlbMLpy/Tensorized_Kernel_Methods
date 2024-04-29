import sys
import unittest

import numpy as np

sys.path.append('./')

from source.features import (
    pure_poli_features, 
    gaussian_kernel_features,
    q2_poli_features,
    q2_fourier_features,
)

class TestFeatures(unittest.TestCase):
    def test_pure_poli_features(self):
        expected = np.array(
            [
                [1, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 2, 4, 8],
            ]
        )
        actual = pure_poli_features(np.arange(3), None, 4)
        self.assertTrue(np.allclose(actual, expected))
    
    def test_gaussian_kernel_features(self):
        expected = np.array(
            [
                [ 0.18846226,  0.05777485],
                [-0.77123494, -0.10430503],
                [-0.52870649,  0.13053439],
            ]
        )
        actual = gaussian_kernel_features(
            np.array([np.pi, 2*np.pi, 3*np.pi]),
            None, 
            order=2,
            lscale=1,
            domain_bound=1,
        )
        self.assertTrue(np.allclose(actual, expected))

    def test_q2_poli_features(self):
        expected = np.array(
            [
                [1, 0],
                [1, 1],
                [1, 16],
                [1, 81]
            ]
        )
        actual = q2_poli_features(
            np.arange(4), 
            q=2,
        )
        self.assertTrue(np.allclose(actual, expected))

    def test_q2_fourier_features(self):
        expected = np.array(
            [
                [ 1.+0.0000000e+00j,  1.+0.0000000e+00j],
                [-1.-1.2246468e-16j, -1.+1.2246468e-16j],
                [ 1.+2.4492936e-16j,  1.-2.4492936e-16j],
                [-1.-3.6739404e-16j, -1.+3.6739404e-16j],
            ]
        )
        actual = q2_fourier_features(
            np.arange(4), 
            q=1,
            m_order=3,
            k_d=3,
            lscale=1
        )
        self.assertTrue(np.allclose(actual, expected))
