from functools import partial
from typing import Optional, Callable

import numpy as np
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from ..features import Feature, PPFeature, fourier_features, pure_poli_features

class RRf(RegressorMixin, BaseEstimator): # Ridge Regression with features learning
    def __init__(
        self,  
        fmaps_list: list[Feature] = [PPFeature(),],
        m_order: int = 2,
        n_epoch: int = 1, 
        alpha: float = 1.0, 
        beta: float = 1.0,
        lambda_reg_type: str = 'l2',
        n_steps_l1: int = 100,
        random_state: Optional[int] = None,
        init_equal_lambda: bool = False,
        positive_lambda: bool = False,
        callback: Optional[Callable] = None,
    ): 
        self.fmaps_list = fmaps_list
        self.m_order = m_order
        self.n_epoch = n_epoch
        self.alpha = alpha
        self.beta = beta
        self.lambda_reg_type = lambda_reg_type
        self.n_steps_l1 = n_steps_l1
        self.random_state = random_state
        self.init_equal_lambda = init_equal_lambda
        self.positive_lambda = positive_lambda
        self.callback = callback
        self._dtype = None
    
    def _prepare_feature_mappings(self):
        mappings = []
        for feature in self.fmaps_list:
            if feature.name == 'ppf':
                mappings.append(partial(pure_poli_features, q=None, order=self.m_order))
            elif feature.name == 'ff':
                mappings.append(
                    partial(fourier_features, q=None, m_order=self.m_order, p_scale=feature.p_scale))
                self._dtype = np.complex128
            else:
                raise ValueError(f'Bad feature_map name = "{feature.name}". See docs.')
        self._dtype = np.float64 if self._dtype is None else self._dtype
        return mappings

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._feature_maps_list = self._prepare_feature_mappings()
        self.weights_, self.lambdas_ = reg_f(
            X, y, self.m_order, self._feature_maps_list, 
            self.n_epoch, self.alpha, self.beta, self.lambda_reg_type, 
            self.n_steps_l1, self.random_state, self.init_equal_lambda, 
            self.positive_lambda, self._dtype, self.callback)
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        X = check_array(X)
        check_is_fitted(self, 'is_fitted_')
        return predict_score(
            X, self.weights_, self.lambdas_, self._feature_maps_list)
    
    def score(self, X, y):
        return r2_score(y, self.predict(X))

### Computational functions ###
from ..features import FeatureMap
from ..matrix_operations import khatri_rao_row
from ..optimization import fista, ls_solution, lsc_solution

def init_weights( 
    n_values: int,
    seed: Optional[int] = None,
    init_equal: bool = False,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    random_state = np.random if seed is None else np.random.RandomState(seed) 
    weights = np.full((n_values, 1), random_state.randn(1))[:, 0] if init_equal else random_state.randn(n_values) 
    return weights.astype(dtype)

def update_feature_weights( 
    x: np.ndarray, 
    y: np.ndarray,
    beta: float,
    weights: np.ndarray, 
    lambdas: np.ndarray,
    feature_maps_list: list[FeatureMap],
    reg_type: str = 'l2',
    n_steps_l1: int = 1000,
    positive: bool = False,
) -> np.ndarray:
    f_mtx = np.empty((x.shape[0], len(lambdas)))
    for p, feature_map in enumerate(feature_maps_list):
        f_mtx[:, p] = _predict_score(x, weights, feature_map)
    if reg_type == 'l2':
        lambdas = ls_solution(f_mtx, y, beta)
    elif reg_type == 'l1':
        lambdas = fista(f_mtx, y, lambdas, beta, n_steps=n_steps_l1)
    elif reg_type == 'fixed_norm':
        lambdas = lsc_solution(f_mtx, y, beta)
    else:
        raise NotImplementedError("Choose 'l1' or 'l2'")
    if positive: # THIS IS PROBABLY NOT RIGHT!!!  
        lambdas = np.maximum(lambdas, 0.0)
    return lambdas

def update_model_weights( 
    x: np.ndarray, 
    y: np.ndarray,
    alp: float,
    weights: np.ndarray, 
    lambdas: np.ndarray,
    feature_maps_list: list[FeatureMap],
) -> np.ndarray:
    Fk = np.zeros((x.shape[0], len(weights)), dtype=weights.dtype)
    for p, feature_map in enumerate(feature_maps_list):
        if lambdas[p]: 
            ft_mtx = feature_map(x[:, 0])
            for q in range(1, x.shape[1]):
                ft_mtx = khatri_rao_row(feature_map(x[:, q]), ft_mtx)
            Fk += lambdas[p] * ft_mtx # Fortran Ordering
    return ls_solution(Fk, y, alp)

def reg_f(
    x: np.ndarray, 
    y: np.ndarray,
    m_order: int,
    fmaps_list: list[FeatureMap],
    n_epoch: int,
    alpha: float,
    beta: float,
    lambda_reg_type: str = 'l2',
    n_steps_l1: int = 100,
    seed: Optional[int] = None,
    init_equal_lambda: bool = False,
    positive_lambda: bool = False,
    dtype: np.dtype = np.float64,
    callback: Optional[Callable] = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """ TODO """
    # Prepare modeling:
    n_fmaps, (_, d_dim) = len(fmaps_list), x.shape
    weights = init_weights(m_order**d_dim, seed, dtype=dtype)
    lambdas = init_weights(n_fmaps, seed, init_equal_lambda, np.float64)
    run_callback(x, y, weights, lambdas, fmaps_list, alpha, beta, callback)
    for _ in range(n_epoch):
        ### TRAIN W ###
        weights = update_model_weights(x, y, alpha, weights, lambdas, fmaps_list)
        run_callback(x, y, weights, lambdas, fmaps_list, alpha, beta, callback)
        ### TRAIN L ###
        lambdas = update_feature_weights(x, y, beta, weights, lambdas, 
            fmaps_list, lambda_reg_type, n_steps_l1, positive_lambda)
        run_callback(x, y, weights, lambdas, fmaps_list, alpha, beta, callback)
        if not lambdas.any(): # if all lambda values are zeros - stop optimization
            break
    return weights, lambdas

def _predict_score(
    x: np.ndarray, 
    weights: np.ndarray, 
    feature_map: FeatureMap
) -> np.ndarray:
    ft_mtx = feature_map(x[:, 0])
    for q in range(1, x.shape[1]):
        ft_mtx = khatri_rao_row(feature_map(x[:, q]), ft_mtx)
    return np.real(ft_mtx.dot(weights))

def predict_score(
    x: np.ndarray, 
    weights: np.ndarray, 
    lambdas: np.ndarray,
    feature_maps_list: list[FeatureMap],
) -> np.ndarray:
    score = np.zeros(x.shape[0])
    for p, feature_map in enumerate(feature_maps_list):
        if lambdas[p]:
            score += lambdas[p] * _predict_score(x, weights, feature_map)
    return score

def run_callback(
        x, 
        y, 
        weights, 
        lambdas, 
        feature_maps_list, 
        alpha,
        beta, 
        callback: Optional[Callable] = None,
):
    if callback:
        y_pred = predict_score(x, weights, lambdas, feature_maps_list)
        callback(y, y_pred, weights, lambdas, feature_maps_list, alpha, beta)
    