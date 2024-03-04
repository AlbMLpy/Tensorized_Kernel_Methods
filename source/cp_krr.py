from typing import Callable, Optional

import numpy as np
from .matrix_operations import khatri_rao_row

FeatureMap = Callable[[np.array], np.array]

def cp_krr(
    x: np.array, 
    y: np.array,
    feature_dim: int,
    feature_map: FeatureMap,
    rank: int,
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
        Weights tensor: n_in_features by feature_dim by cp-rank
        
    """

    # Initialize model weights:
    _, d_dim = x.shape
    random_state = np.random if seed is None else np.random.RandomState(seed)
    weights = random_state.randn(d_dim, feature_dim, rank)

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
            A = Fk.T.dot(Fk) + reg_value * np.kron(hadamard_gram, np.eye(feature_dim))
            wk = weights[k] = np.linalg.solve(A, b).reshape(feature_dim, rank)
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
