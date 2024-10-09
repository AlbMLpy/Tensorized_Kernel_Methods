import sys
import unittest
from functools import partial

import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

sys.path.append('./')

from source.jax.model_functionality import (
    init_weights, 
    get_fw_hadamard_mtx,
    get_ww_hadamard_mtx,
    get_updated_als_factor,
)
from source.jax.features import pure_poli_features, ppf_q2


class TestModelFunctionality(unittest.TestCase):
    def test_init_weights_non_quant(self):
        m_order, rank, d_dim, q_base = 13, 5, 4, None
        temp, _ = init_weights(m_order, rank, d_dim, q_base)
        
        expected = jnp.array([d_dim, m_order, rank])
        actual = jnp.array(temp.shape)
        self.assertTrue(jnp.allclose(actual, expected))
    
    def test_init_weights_quant(self):
        m_order, rank, d_dim, q_base = 16, 5, 4, 2
        temp, _ = init_weights(m_order, rank, d_dim, q_base)
        
        expected = jnp.array(
            [d_dim*int(jnp.log2(m_order)), q_base, rank])
        actual = jnp.array(temp.shape)
        self.assertTrue(jnp.allclose(actual, expected))

    def test_init_weights_bad_m_order(self):
        # Quantized setting:
        m_order, rank, d_dim, q_base = 13, 5, 4, 2
        with self.assertRaises(ValueError):
            init_weights(m_order, rank, d_dim, q_base)

    def test_get_updated_als_factor(self):
        n, f_dim = 3, 2
        fk_mtx = jnp.ones((n, f_dim))
        fw_hadamard = jnp.array(
             [[1.0, 2], [2, 4], [4, 8]]
        )
        ww_hadamard = jnp.array([[1.0, 3], [3, 5]])
        y = jnp.array([1.0, 0, 1])
        alpha = 1.0
        
        expected = jnp.array(
            [
                [0.03846154, 0.03846154],
                [0.03846154, 0.03846154]
            ]
        )
        actual = get_updated_als_factor(
            fk_mtx, 
            fw_hadamard,
            ww_hadamard, 
            y, 
            alpha,
        )
        self.assertTrue(jnp.allclose(actual, expected))

    def test_get_fw_hadamard_mtx_quant(self):
        x = jnp.array(
            [
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
            ]
        )
        k_d = 2
        weights = jnp.array(
            [
                [[1, 2], [2, 3]], 
                [[0, 1], [1, 0]],
                [[1, 2], [2, 3]], 
                [[0, 1], [1, 0]]
            ]
        )
        feature_map = ppf_q2

        expected = jnp.array(
            [
                [9., 25],
                [400., 64.],
                [3969., 121.],
                [20736., 196.]
            ]
        )
        actual = get_fw_hadamard_mtx(x, k_d, weights, feature_map)
        self.assertTrue(jnp.allclose(actual, expected))

    def test_get_fw_hadamard_mtx_non_quant(self):
        x = jnp.array(
            [
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
            ]
        )
        k_d = 1
        weights = jnp.array(
            [
                [[1, 2], [2, 3], [3, 4]], 
                [[0, 1], [1, 0], [1, 1]]
            ]
        )
        feature_map = partial(pure_poli_features, order=3)

        expected = jnp.array(
            [
                [  12.,   18.],
                [ 102.,  120.],
                [ 408.,  470.],
                [1140., 1326.]
            ]
        )
        actual = get_fw_hadamard_mtx(x, k_d, weights, feature_map)
        self.assertTrue(jnp.allclose(actual, expected))

    def test_get_ww_hadamard_mtx(self):
        weights = jnp.array(
            [
                [[1, 2], [2, 3], [3, 4]], 
                [[0, 1], [1, 0], [1, 1]]
            ]
        )

        expected = jnp.array([[28., 20.], [20., 58.]])
        actual = get_ww_hadamard_mtx(weights)
        self.assertTrue(jnp.allclose(actual, expected))
