import sys
import unittest

import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

sys.path.append('./')

from source.jax.matrix_operations import khatri_rao_row

class TestFeatures(unittest.TestCase):
    def test_khatri_rao_row(self):
        # prepare the data:
        a = jnp.arange(1, 5).reshape(2, 2)
        b = jnp.arange(1, 7).reshape(2, 3)

        expected = jnp.array(
            [
                [ 1,  2,  3,  2,  4,  6],
                [12, 15, 18, 16, 20, 24],
            ]
        )
        actual = khatri_rao_row(a, b)
        self.assertTrue(jnp.allclose(actual, expected))
