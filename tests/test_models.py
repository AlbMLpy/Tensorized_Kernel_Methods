import sys
import unittest
from typing import Callable

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler

sys.path.append('./')

from source.models.RRf import RRf
from source.models.CPR import CPR
from source.models.QCPR import QCPR
from source.models.QCPRf import QCPRf
from source.features import PPFeature, FFeature
from source.loss import ls_l2w_l2l_loss, ls_loss, l2_reg, ls_l2_loss

def mse_l2wl_loss(
    y: np.ndarray, 
    y_pred: np.ndarray, 
    weights: np.ndarray, 
    lambdas: np.ndarray,
    alpha: float,
    beta: float,
) -> float:
    return ls_loss(y, y_pred) + alpha * l2_reg(weights) + beta * l2_reg(lambdas)

def prepare_callback_no_fl():
    def callback_function(
        y: np.ndarray, 
        y_pred: np.ndarray, 
        k_d: int,
        weights: np.ndarray,
        feature_map, 
        alpha,
        y_yp,
        *args,
        **kwargs,
    ):
        if not hasattr(callback_function, 'data'):
            callback_function.data = [] 
        value = ls_l2_loss(y, y_pred, weights, alpha)
        callback_function.data.append(value)
    return callback_function

def prepare_callback_fl(loss_funct: Callable):
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
        value = loss_funct(y, y_pred, weights, lambdas, alpha, beta)
        callback_function.data.append(value)
    return callback_function

def prepare_callback_fl_1(loss_funct: Callable):
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
        value = loss_funct(y, y_pred, weights, lambdas, alpha, beta)
        callback_function.data.append(value)
    return callback_function

class TestModels(unittest.TestCase):
    def setUp(self) -> None:
        diabetes = load_diabetes()
        x, y = diabetes.data, diabetes.target
        self.x = MinMaxScaler().fit_transform(x)
        self.y = (y - y.mean()) / y.std()
        self.model_params = dict(rank=8, m_order=4, n_epoch=10, alpha=0.001, random_state=0)

    def test_cpr_loss(self):
        callback_function = prepare_callback_no_fl()
        model_params = dict(callback=callback_function) | self.model_params
        model = CPR(**model_params)
        model.fit(self.x, self.y) 

        expected = sorted(callback_function.data, reverse=True)
        actual = callback_function.data
        self.assertTrue(np.allclose(actual, expected))

    def test_qcpr_loss(self):
        callback_function = prepare_callback_no_fl()
        model_params = dict(callback=callback_function) | self.model_params
        model = QCPR(**model_params)
        model.fit(self.x, self.y) 

        expected = sorted(callback_function.data, reverse=True)
        actual = callback_function.data
        self.assertTrue(np.allclose(actual, expected))

    def test_qcprf_loss(self):        
        callback_function = prepare_callback_fl(ls_l2w_l2l_loss)
        model_params = (
            self.model_params 
            | dict(callback=callback_function, beta=0.001, 
                fmaps_list=[PPFeature(), FFeature(p_scale=2)])
        ) 
        model = QCPRf(**model_params)
        model.fit(self.x, self.y) 

        expected = sorted(callback_function.data, reverse=True)
        actual = callback_function.data
        self.assertTrue(np.allclose(actual, expected))

    def test_rrf_loss(self):    
        callback_function = prepare_callback_fl_1(mse_l2wl_loss)
        model_params = dict(
            m_order=3, n_epoch=10, alpha=0.001, random_state=0,
            callback=callback_function, beta=1, lambda_reg_type='l2',
            fmaps_list=[FFeature(p_scale=3), FFeature(p_scale=2)]
        )
        model = RRf(**model_params)
        model.fit(self.x[:, :5], self.y) 
        loss_value = callback_function.data[1::2]

        expected = sorted(loss_value, reverse=True)
        actual = loss_value
        self.assertTrue(np.allclose(actual, expected))
