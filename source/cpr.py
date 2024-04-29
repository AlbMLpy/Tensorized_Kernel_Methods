from typing import Optional, Callable

import numpy as np

from .features import FeatureMap
from .model_functionality import (
    init_weights,
    get_fw_hadamard_mtx,
    get_ww_hadamard_mtx,
    update_weights,
    run_callback,
)

Q_BASE = 2

def cpr(
    x: np.ndarray, 
    y: np.ndarray,
    quantized: bool, 
    m_order: int,
    feature_map: FeatureMap,
    rank: int,
    init_type: str,
    n_epoch: int,
    alpha: float,
    seed: Optional[int] = None,
    dtype: np.dtype = np.float64,
    callback: Optional[Callable] = None,
) -> np.ndarray:
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
    q_base = Q_BASE if quantized else None
    weights, k_d = init_weights(m_order, rank, x.shape[-1], q_base, init_type, seed, dtype)
    fw_hadamard = get_fw_hadamard_mtx(x, k_d, weights, feature_map, dtype)
    ww_hadamard = get_ww_hadamard_mtx(weights, dtype)
    run_callback(x, y, alpha, k_d, weights, feature_map, callback)
    for _ in range(n_epoch):
        weights = update_weights(
            x, y, alpha, k_d, weights, feature_map, fw_hadamard, ww_hadamard)
        run_callback(x, y, alpha, k_d, weights, feature_map, callback)
    return weights, k_d

### See Sandbox!!!
def weights_to_quantized_4d_tensor(weights: np.ndarray) -> np.ndarray:
    pass
