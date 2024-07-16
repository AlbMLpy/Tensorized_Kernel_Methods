import sys
import unittest

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler

sys.path.append('./')

from source.models.RRf import RRf
from source.models.CPR import CPR
from source.models.QCPR import QCPR
from source.models.QCPRf import QCPRf
from source.features import PPFeature, FFeature
from source.general_functions import prepare_callback_mse_wl2
from source.loss import mse_l2w_l2l_loss, mse_loss, l2_reg


class TestModels(unittest.TestCase):
    def setUp(self) -> None:
        diabetes = load_diabetes()
        x, y = diabetes.data, diabetes.target
        self.x = MinMaxScaler().fit_transform(x)
        self.y = (y - y.mean()) / y.std()

    def test_cpr_loss(self):
        callback_function = prepare_callback_mse_wl2()
        model_params = dict(
            rank=8,
            m_order=4,
            n_epoch=20,
            alpha=0.001,
            random_state=0,
            callback=callback_function,
        )
        model = CPR(**model_params)
        model.fit(self.x, self.y) 

        expected = sorted(callback_function.data, reverse=True)
        actual = callback_function.data
        self.assertTrue(np.allclose(actual, expected))

    def test_qcpr_loss(self):
        callback_function = prepare_callback_mse_wl2()
        model_params = dict(
            rank=8,
            m_order=4,
            n_epoch=20,
            alpha=0.001,
            random_state=0,
            callback=callback_function,
        )
        model = QCPR(**model_params)
        model.fit(self.x, self.y) 

        expected = sorted(callback_function.data, reverse=True)
        actual = callback_function.data
        self.assertTrue(np.allclose(actual, expected))

    def test_qcprf_loss(self):
        def prepare_callback():
            def callback_function(
                y: np.ndarray, 
                y_pred: np.ndarray, 
                k_d: int,
                weights: np.ndarray,
                lambdas: np.ndarray,
                feature_maps_list, 
                alpha,
                beta, 
                *args,
                **kwargs,
            ):
                if not hasattr(callback_function, 'data'):
                    callback_function.data = []   
                value = mse_l2w_l2l_loss(y, y_pred, weights, lambdas, alpha, beta)
                callback_function.data.append(value)
            return callback_function
        
        callback_function = prepare_callback()
        model_params = dict(
            rank=8,
            m_order=4,
            fmaps_list=[
                PPFeature(), 
                FFeature(p_scale=1), 
            ],
            n_epoch=5,
            alpha=0.001,
            beta=0.001,
            random_state=0,
            callback=callback_function,
        )
        model = QCPRf(**model_params)
        model.fit(self.x, self.y) 

        expected = sorted(callback_function.data, reverse=True)
        actual = callback_function.data
        self.assertTrue(np.allclose(actual, expected))

    def test_rrf_loss(self):
        def prepare_callback():
            def mse_l2wl_loss(
                y: np.ndarray, 
                y_pred: np.ndarray, 
                weights: np.ndarray, 
                lambdas: np.ndarray,
                alpha: float,
                beta: float,
            ) -> float:
                return mse_loss(y, y_pred) + alpha * l2_reg(weights) + beta * l2_reg(lambdas)
            
            def callback_function(
                y: np.ndarray, 
                y_pred: np.ndarray, 
                weights: np.ndarray,
                lambdas: np.ndarray,
                feature_maps_list, 
                alpha,
                beta, 
                *args,
                **kwargs,
            ):
                if not hasattr(callback_function, 'data'):
                    callback_function.data = []   
                value = mse_l2wl_loss(y, y_pred, weights, lambdas, alpha, beta)
                callback_function.data.append(value)
            return callback_function
        
        callback_function = prepare_callback()
        model_params = dict(
            m_order=3,
            fmaps_list=[
                FFeature(p_scale=2), 
                FFeature(p_scale=10), 
                FFeature(p_scale=30), 
                FFeature(p_scale=70), 
            ],
            n_epoch=5,
            alpha=0.001,
            beta=1,
            lambda_reg_type='l2',
            random_state=0,
            callback=callback_function,
        )
        model = RRf(**model_params)
        model.fit(self.x[:, :5], self.y) 

        expected = sorted(callback_function.data, reverse=True)
        actual = callback_function.data
        self.assertTrue(np.allclose(actual, expected))
