from typing import Optional, Callable

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

def prepare_system(
    fk_mtx: np.array, 
    fw_hadamard: np.array,
    y: np.array,
):
    Fk = khatri_rao_row(fw_hadamard, fk_mtx) # Fortran Ordering
    return Fk.T.conj().dot(Fk), Fk.T.conj().dot(y)

def get_updated_als_factor(
    fk_mtx: np.array, 
    fw_hadamard: np.array,
    ww_hadamard: np.array,
    y: np.array,
    alpha: float,
) -> np.array:
    """ 
    Solve custom linear system of equations.
    TODO
    """
    (_, f_dim), (rank, _) = fk_mtx.shape, ww_hadamard.shape
    A, b = prepare_system(fk_mtx, fw_hadamard, y)
    if alpha:
        A += alpha * np.kron(ww_hadamard, np.eye(f_dim)) # Fortran Ordering
    return np.linalg.solve(A, b).reshape(f_dim, rank, order='F') # Fortran Ordering

def update_weights(
    x: np.array, 
    y: np.array,
    alpha: float,
    weights: np.array,
    feature_map: FeatureMap,
    fw_hadamard: np.array,
    ww_hadamard: np.array,
) -> np.array:
    d_dim, _, _ = weights.shape
    for k in range(d_dim):
        wk, fk_mtx = weights[k], feature_map(x[:, k])
        # Preprocess:
        fw_hadamard /= fk_mtx.dot(wk) # remove k-th factor
        ww_hadamard /= wk.T.conj().dot(wk) # remove k-th factor
        # Calculate A, b and solve linear system:
        wk = weights[k] = get_updated_als_factor(
            fk_mtx, fw_hadamard, ww_hadamard, y, alpha)
        # Postprocess:
        fw_hadamard *= fk_mtx.dot(wk)
        ww_hadamard *= wk.T.conj().dot(wk)
    return weights

def cp_krr(
    x: np.array, 
    y: np.array,
    m_order: int,
    feature_map: FeatureMap,
    rank: int,
    init_type: str,
    n_epoch: int,
    alpha: float,
    seed: Optional[int] = None,
    callback: Optional[Callable] = None,
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
    ...
    callback : Optional[Callable] = None
        Function is called before training and after every epoch of training. 
        callback should have the following layout: callback(y, y_pred, weights, **kwargs).
    
    Returns
    -------
    output : numpy.ndarray[:, :, :]
        Weights tensor: n_in_features by m_order by cp-rank
        
    """
    _, d_dim = x.shape
    weights = init_weights(m_order, rank, d_dim, init_type, seed=seed)
    fw_hadamard, ww_hadamard = get_hadamard_matrices(x, weights, feature_map)
    run_callback(x, y, alpha, weights, feature_map, callback)
    for _ in range(n_epoch):
        weights = update_weights(
            x, y, alpha, weights, feature_map, fw_hadamard, ww_hadamard)
        run_callback(x, y, alpha, weights, feature_map, callback)
    return weights

def predict_score(x: np.array, weights: np.array, feature_map: FeatureMap) -> np.array:
    (n_samples, _), (d_dim, _, rank) = x.shape, weights.shape
    score = np.ones((n_samples, rank))
    for k in range(d_dim): 
        score *= feature_map(x[:, k]).dot(weights[k])
    return np.sum(score, 1)

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
