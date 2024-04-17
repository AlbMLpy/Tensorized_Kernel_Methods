from typing import Optional, Callable
from itertools import product

import numpy as np

from .features import FeatureMap
from .cp_krr import get_updated_als_factor

Q_BASE = 2

def init_quantized_weights(
    m_order: int, 
    rank: int, 
    d_dim: int, 
    q_base: int = 2, 
    init_type: str = 'k_mtx', 
    seed: Optional[int] = None,
    dtype: np.dtype = np.float64,
) -> np.array:
    """ 
    Random initialization of model parameters. 
    TODO
    """
    if m_order & (m_order - 1):
        raise ValueError(f"m_order should be a power of 2, but it is {m_order}. ")
    k_d = int(np.emath.logn(q_base, m_order)) # m_order = q_base^(k_d)
    random_state = np.random if seed is None else np.random.RandomState(seed) 
    weights = random_state.randn(d_dim, k_d, q_base, rank)
    if init_type == 'k_mtx': # Matrix weights[k][l][:, :] is normalized
        weights /= np.linalg.norm(weights, ord=2, axis=(2, 3), keepdims=True)
    elif init_type == 'kj_vec': # Vector weights[k][l][:][j] is normalized
        weights /= np.linalg.norm(weights, ord=2, axis=2, keepdims=True)
    else:
        raise ValueError(f'Bad init_type = {init_type}. See docs.')
    return weights.astype(dtype)

def get_fw_hadamard_mtx(
    x: np.array, 
    weights: np.array, 
    feature_map: FeatureMap,
    dtype: np.dtype = np.float64,
) -> np.array:
    d_dim, k_d, _, _ = weights.shape
    fw_hadamard = 1.0
    for k, q in product(range(d_dim), range(k_d)):
        fw_hadamard *= feature_map(x[:, k], q).dot(weights[k, q])
    return fw_hadamard.astype(dtype)

def get_ww_hadamard_mtx(
    weights: np.array, 
    dtype: np.dtype = np.float64,
) -> np.array:
    d_dim, k_d, _, _ = weights.shape
    ww_hadamard = 1.0
    for k, q in product(range(d_dim), range(k_d)):
        wk = weights[k, q]
        ww_hadamard *= wk.T.dot(wk)
    return ww_hadamard.astype(dtype)

def update_quantized_weights(
    x: np.array, 
    y: np.array,
    alpha: float,
    weights: np.array,
    feature_map: FeatureMap,
    fw_hadamard: np.array,
    ww_hadamard: np.array,
) -> np.array:
    d_dim, k_d, _, _ = weights.shape
    for k, q in product(range(d_dim), range(k_d)):
        wk, fk_mtx = weights[k, q], feature_map(x[:, k], q)
        # Preprocess:
        fw_hadamard /= fk_mtx.dot(wk) # remove k-th factor
        ww_hadamard /= wk.T.conj().dot(wk) # remove k-th factor
        # Calculate A, b and solve linear system:
        wk = weights[k, q] = get_updated_als_factor(
            fk_mtx, fw_hadamard, ww_hadamard, y, alpha)
        # Postprocess:
        fw_hadamard *= fk_mtx.dot(wk)
        ww_hadamard *= wk.T.conj().dot(wk)
    return weights

def qcp_krr(
    x: np.array, 
    y: np.array,
    m_order: int,
    feature_map: FeatureMap,
    rank: int,
    init_type: str,
    n_epoch: int,
    alpha: float,
    seed: Optional[int] = None,
    dtype: np.dtype = np.float64,
    callback: Optional[Callable] = None,
) -> np.array:
    _, d_dim = x.shape
    weights = init_quantized_weights(m_order, rank, d_dim, Q_BASE, init_type, seed, dtype)
    fw_hadamard = get_fw_hadamard_mtx(x, weights, feature_map, dtype)
    ww_hadamard = get_ww_hadamard_mtx(weights, dtype)
    run_callback(x, y, alpha, weights, feature_map, callback)
    for _ in range(n_epoch):
        weights = update_quantized_weights(
            x, y, alpha, weights, feature_map, fw_hadamard, ww_hadamard)
        run_callback(x, y, alpha, weights, feature_map, callback)
    return weights

def predict_score(x: np.array, weights: np.array, feature_map: FeatureMap) -> np.array:
    (n_samples, _), (d_dim, k_d, _, rank) = x.shape, weights.shape
    score = np.ones((n_samples, rank), dtype=weights.dtype)
    for k, q in product(range(d_dim), range(k_d)):
        score *= feature_map(x[:, k], q).dot(weights[k, q])
    return np.real(np.sum(score, 1))

def run_callback(
        x: np.array, 
        y: np.array, 
        alpha: float, 
        weights: np.array,  
        feature_map: FeatureMap, 
        callback: Optional[Callable] = None,
):
    if callback:
        y_pred = predict_score(x, weights, feature_map)
        callback(y, y_pred, weights, alpha=alpha)
