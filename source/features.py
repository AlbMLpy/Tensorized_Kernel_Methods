import numpy as np

def pure_poli_features(x: np.array, order: int) -> np.array:
    """ Calculate pure polinomial features matrix for x """
    return np.power(x[:, None], np.arange(order))
