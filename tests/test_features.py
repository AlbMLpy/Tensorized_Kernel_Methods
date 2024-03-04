import sys
import unittest

import numpy as np

sys.path.append('./')

from source.features import pure_poli_features

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
