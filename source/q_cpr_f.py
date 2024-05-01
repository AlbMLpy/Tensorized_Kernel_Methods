from typing import Optional, Callable

import numpy as np

from .features import FeatureMap
from .matrix_operations import khatri_rao_row
from .model_functionality import (
    init_weights,
    get_fw_hadamard_mtx,
    get_ww_hadamard_mtx,
    predict_score as predict_score_fm,
)
from .cpr import Q_BASE

def init_feature_weights( 
    n_values: int,
    seed: Optional[int] = None,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    random_state = np.random if seed is None else np.random.RandomState(seed) 
    weights = random_state.randn(n_values)
    return weights.astype(dtype)

def precalculate_step(
    x: np.ndarray,
    k_d: int,
    weights: np.ndarray,
    feature_maps_list: list[FeatureMap],
    dtype: np.dtype = np.float64
) -> tuple[np.ndarray, np.ndarray]:
    shape = (len(feature_maps_list), x.shape[0], weights.shape[-1])
    fwh_matrices = np.empty(shape, dtype=weights.dtype)
    for p, feature_map in enumerate(feature_maps_list):
        fwh_matrices[p] = get_fw_hadamard_mtx(x, k_d, weights, feature_map, dtype)
    ww_hadamard = get_ww_hadamard_mtx(weights, dtype)
    return fwh_matrices, ww_hadamard

def update_quantized_weights(
    x: np.ndarray, 
    y: np.ndarray,
    alpha: float,
    k_d: int,
    weights: np.ndarray,
    lambdas: np.ndarray,
    feature_maps_list: list[FeatureMap],
    fwh_matrices: np.ndarray,
    ww_hadamard: np.ndarray,
) -> np.ndarray:
    (n, _), (q_base, rank) = x.shape, weights.shape[-2:]
    for ind in range(weights.shape[0]):
        k, q = divmod(ind, k_d)
        wk, Fk = weights[ind], np.zeros((n, q_base*rank), dtype=weights.dtype)
        # Preprocess:
        ww_hadamard /= wk.T.conj().dot(wk)
        for p, feature_map in enumerate(feature_maps_list):
            fk_mtx = feature_map(x[:, k], q)
            fwh_matrices[p] /= fk_mtx.dot(wk) # remove k-th factor
            Fk += lambdas[p] * khatri_rao_row(fwh_matrices[p], fk_mtx) # Fortran Ordering
        # Calculate A, b and solve linear system:
        A, b = Fk.T.conj().dot(Fk), Fk.T.conj().dot(y)
        if alpha:
            A += alpha * np.kron(ww_hadamard, np.eye(q_base)) # Fortran Ordering
        wk = np.linalg.solve(A, b).reshape(q_base, rank, order='F') # Fortran Ordering
        # Postprocess:
        ww_hadamard *= wk.T.conj().dot(wk)
        for p, feature_map in enumerate(feature_maps_list):
            fk_mtx = feature_map(x[:, k], q)
            fwh_matrices[p] *= fk_mtx.dot(wk)
        
        weights[ind] = wk
    return weights

def update_feature_weights( 
    x: np.ndarray, 
    y: np.ndarray,
    beta: float,
    k_d: int,
    weights: np.ndarray, 
    lambdas: np.ndarray,
    feature_maps_list: list[FeatureMap],
) -> np.ndarray:
    n_fmaps = len(lambdas)
    f_mtx = np.empty((x.shape[0], n_fmaps))
    for p, feature_map in enumerate(feature_maps_list):
        f_mtx[:, p] = predict_score_fm(x, k_d, weights, feature_map)
    a_mtx, b = f_mtx.T.conj().dot(f_mtx), f_mtx.T.conj().dot(y)
    if beta:
        a_mtx += beta * np.eye(n_fmaps)
    return np.linalg.solve(a_mtx, b)
    
def q_cpr_f(
    x: np.ndarray, 
    y: np.ndarray,
    m_order: int,
    fmaps_list: list[FeatureMap],
    rank: int,
    init_type: str,
    n_epoch: int,
    alpha: float,
    beta: float,
    seed: Optional[int] = None,
    dtype: np.dtype = np.float64,
    callback: Optional[Callable] = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """ TODO """

    n_fmaps, (_, d_dim) = len(fmaps_list), x.shape
    weights, k_d = init_weights(m_order, rank, d_dim, Q_BASE, init_type, seed, dtype)
    lambdas = init_feature_weights(n_fmaps, seed, dtype)
    fwh_matrices, ww_hadamard = precalculate_step(x, k_d, weights, fmaps_list, dtype)
    run_callback(x, y, k_d, weights, lambdas, fmaps_list, alpha, callback)
    for _ in range(n_epoch):
        weights = update_quantized_weights(
            x, y, alpha, k_d, weights, lambdas, fmaps_list, fwh_matrices, ww_hadamard)
        lambdas = update_feature_weights(x, y, beta, k_d, weights, lambdas, fmaps_list)
        run_callback(x, y, k_d, weights, lambdas, fmaps_list, alpha, callback)
    return weights, lambdas, k_d

def predict_score(
    x: np.ndarray, 
    k_d: int,
    weights: np.ndarray, 
    lambdas: np.ndarray,
    feature_maps_list: list[FeatureMap],
) -> np.ndarray:
    score = 0.0
    for p, feature_map in enumerate(feature_maps_list):
        score += lambdas[p] * predict_score_fm(x, k_d, weights, feature_map)
    return score

def run_callback(
        x, 
        y, 
        k_d: int,
        weights, 
        lambdas, 
        feature_maps_list, 
        alpha, 
        callback: Optional[Callable] = None,
):
    if callback:
        y_pred = predict_score(x, k_d, weights, lambdas, feature_maps_list)
        callback(y, y_pred, weights, alpha=alpha)
