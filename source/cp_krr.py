from typing import Optional

import numpy as np

from .features import FeatureMap
from .matrix_operations import khatri_rao_row

def init_weights(
    m_order: int, 
    rank: int, 
    d_dim: int, 
    init_type: str = 'k_mtx', 
    seed: Optional[int] = None
) -> np.array:
    """ 
    Random initialization of model parameters. 
    TODO
    """
    random_state = np.random if seed is None else np.random.RandomState(seed) 
    weights = random_state.randn(d_dim, m_order, rank)
    if init_type == 'k_mtx': # Matrix weights[k][:, :] is normalized
        weights /= np.linalg.norm(weights, ord=2, axis=(1, 2), keepdims=True)
    elif init_type == 'kj_vec': # Vector weights[k][:][j] is normalized
        weights /= np.linalg.norm(weights, ord=2, axis=1, keepdims=True)
    else:
        raise ValueError(f'Bad init_type = {init_type}. See docs.')
    return weights

def get_hadamard_matrices(
    x: np.array, 
    weights: np.array, 
    feature_map: FeatureMap
) -> tuple[np.array, np.array]:
    """ 
    Preprocessing: Calculate full features-parameters multiplication
    TODO
    """
    d_dim, _, _ = weights.shape
    fw_hadamard, ww_hadamard = 1.0, 1.0
    for k in range(d_dim):
        wk = weights[k]
        fw_hadamard *= feature_map(x[:, k]).dot(wk)
        ww_hadamard *= wk.T.dot(wk)
    return fw_hadamard, ww_hadamard

def get_updated_als_factor(
    fk_mtx: np.array, 
    fw_hadamard: np.array,
    ww_hadamard: np.array,
    y: np.array,
    reg_value: float,
    m_order: int,
) -> np.array:
    """ 
    Solve custom linear system of equations.
    TODO
    """
    Fk = khatri_rao_row(fw_hadamard, fk_mtx) # Fortran Ordering
    b = Fk.T.dot(y)
    A = Fk.T.dot(Fk) 
    if reg_value:
        A += reg_value * np.kron(ww_hadamard, np.eye(m_order)) # Fortran Ordering
    rank, _ = ww_hadamard.shape
    return np.linalg.solve(A, b).reshape(m_order, rank, order='F') # Fortran Ordering

def cp_krr(
    x: np.array, 
    y: np.array,
    m_order: int,
    feature_map: FeatureMap,
    rank: int,
    init_type: str,
    n_epoch: int,
    reg_value: float,
    seed: Optional[int] = None
) -> np.array:
    """ 
    Train Tensor-Kernel Ridge Regression model (CP).

    References: 
        "Large-Scale Learning with Fourier Features and Tensor Decompositions", Wesel, Batselier.
    
    Parameters
    ----------

    x : numpy.ndarray[:, :]
        Input data X: n_samples by n_in_features
    y : numpy.ndarray[:]
        Target values y: n_samples
    
    Returns
    -------
    output : numpy.ndarray[:, :, :]
        Weights tensor: n_in_features by m_order by cp-rank
        
    """
    _, d_dim = x.shape
    weights = init_weights(m_order, rank, d_dim, init_type, seed=seed)
    fw_hadamard, ww_hadamard = get_hadamard_matrices(x, weights, feature_map)
    for _ in range(n_epoch):
        for k in range(d_dim):
            wk, fk_mtx = weights[k], feature_map(x[:, k])
            # Preprocess:
            fw_hadamard /= fk_mtx.dot(wk) # remove k-th factor
            ww_hadamard /= wk.T.dot(wk) # remove k-th factor
            # Calculate A, b and solve linear system:
            wk = weights[k] = get_updated_als_factor(
                fk_mtx, fw_hadamard, ww_hadamard, y, reg_value, m_order)
            # Postprocess:
            fw_hadamard *= fk_mtx.dot(wk)
            ww_hadamard *= wk.T.dot(wk)
    return weights

def predict_score(x: np.array, weights: np.array, feature_map: FeatureMap) -> np.array:
    (n_samples, d_dim), rank = x.shape, weights.shape[2]
    score = np.ones((n_samples, rank))
    for k in range(d_dim): 
        score *= feature_map(x[:, k]).dot(weights[k])
    return np.sum(score, 1)
