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
    # Preprocessing: Calculate full features-parameters multiplication: 
    hadamard_feat_param = 1.0
    hadamard_gram = 1.0
    for k in range(d_dim):
        wk = weights[k]
        hadamard_feat_param *= feature_map(x[:, k]).dot(wk)
        hadamard_gram *= wk.T.dot(wk)
    # Training:
    for _ in range(n_epoch):
        for k in range(d_dim):
            # Preprocess:
            wk = weights[k]
            fk_mtx = feature_map(x[:, k])
            hadamard_feat_param /= fk_mtx.dot(wk) # remove k-th factor
            hadamard_gram /= wk.T.dot(wk) # remove k-th factor
            # Calculate A, b and solve linear system:
            Fk = khatri_rao_row(fk_mtx, hadamard_feat_param)
            b = Fk.T.dot(y)
            A = Fk.T.dot(Fk) + reg_value * np.kron(hadamard_gram, np.eye(m_order))
            wk = weights[k] = np.linalg.solve(A, b).reshape(m_order, rank)
            # Postprocess:
            hadamard_feat_param *= fk_mtx.dot(wk)
            hadamard_gram *= wk.T.dot(wk)
    return weights

def predict_score(x: np.array, weights: np.array, feature_map: FeatureMap) -> np.array:
    (n_samples, d_dim), rank = x.shape, weights.shape[2]
    score = np.ones((n_samples, rank))
    for k in range(d_dim): 
        score *= feature_map(x[:, k]).dot(weights[k])
    return np.sum(score, 1)
