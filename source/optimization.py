import numpy as np

def shrinkage_operator(x: np.ndarray, alpha: float) -> np.ndarray:
    return np.sign(x) * np.maximum((np.abs(x) - alpha), 0)

def ista( 
    x: np.ndarray, 
    y: np.ndarray,
    w: np.ndarray,
    beta: float,
    n_steps: int,
) -> np.ndarray:
    a_mtx, b = x.T.conj().dot(x), x.T.conj().dot(y)
    ss = 1 / np.linalg.norm(a_mtx, ord=2)
    for _ in range(n_steps):
        w = shrinkage_operator(w - ss*(a_mtx.dot(w) - b), beta * ss)
    return w

def fista( 
    x: np.ndarray, 
    y: np.ndarray,
    w: np.ndarray,
    beta: float,
    n_steps: int,
) -> np.ndarray:
    a_mtx, b = x.T.conj().dot(x), x.T.conj().dot(y)
    yk, tk_prev = 1*w, 1
    ss = 1 / np.linalg.norm(a_mtx, ord=2)
    for _ in range(n_steps):
        w = shrinkage_operator(yk - ss*(a_mtx.dot(yk) - b), beta * ss)
        tk = 0.5 * (1 + np.sqrt(1 + 4*tk_prev**2))
        yk = w + (tk_prev - 1) / tk * (w - yk)
        tk_prev = tk
    return w

def cd_ista(
    x: np.ndarray, 
    y: np.ndarray,
    w: np.ndarray,
    beta: float,
    n_steps: int = 1,
) -> np.ndarray:
    a_mtx, b = x.T.conj().dot(x), x.T.conj().dot(y)
    _, n_f = x.shape
    inxs = np.arange(n_f)
    nms = (x * x).sum(axis=0)
    for _ in range(n_steps):
        for i in range(len(w)):
            a_i = a_mtx[i]
            w[i] = 1 / nms[i] * shrinkage_operator(
                b[i] - (a_i * np.where(inxs == i, 0, 1)).dot(w), beta)
    return w

def ls_solution(
    x: np.ndarray, 
    y: np.ndarray, 
    beta: float, 
):
    a_mtx, b = x.T.conj().dot(x), x.T.conj().dot(y)
    if beta: 
        a_mtx += beta * np.eye(a_mtx.shape[0])
    return np.linalg.solve(a_mtx, b)
