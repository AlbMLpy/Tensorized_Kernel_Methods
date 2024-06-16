import numpy as np

from .model_functionality import get_ww_hadamard_mtx

def mse_loss(y: np.ndarray, y_pred: np.ndarray) -> float:
    return 0.5 * np.linalg.norm(y - y_pred, ord=2)**2

def l2_reg_cp(weights: np.ndarray) -> float:
    ww_hadamard = get_ww_hadamard_mtx(weights, weights.dtype)
    return 0.5 * np.sum(np.real(ww_hadamard))

def l2_reg(x: np.ndarray) -> float:
    return 0.5 * np.linalg.norm(x.flatten(), ord=2)**2

def l1_reg(x: np.ndarray) -> float:
    return np.linalg.norm(x.flatten(), ord=1)

def mse_l2_loss(
    y: np.ndarray, 
    y_pred: np.ndarray, 
    weights: np.ndarray, 
    alpha: float,
) -> float:
    return mse_loss(y, y_pred) + alpha * l2_reg_cp(weights)

def mse_l1(
    y: np.ndarray, 
    y_pred: np.ndarray, 
    weights: np.ndarray, 
    alpha: float,
) -> float:
    return mse_loss(y, y_pred) + alpha * l1_reg(weights)

def mse_l2w_l2l_loss(
    y: np.ndarray, 
    y_pred: np.ndarray, 
    weights: np.ndarray, 
    lambdas: np.ndarray,
    alpha: float,
    beta: float,
) -> float:
    return mse_loss(y, y_pred) + alpha * l2_reg_cp(weights) + beta * l2_reg(lambdas)

def mse_l2w_l1l_loss(
    y: np.ndarray, 
    y_pred: np.ndarray, 
    weights: np.ndarray, 
    lambdas: np.ndarray,
    alpha: float,
    beta: float,
) -> float:
    return mse_loss(y, y_pred) + alpha * l2_reg_cp(weights) + beta * l1_reg(lambdas)
