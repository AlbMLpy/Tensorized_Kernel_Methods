from typing import Callable
from collections import namedtuple

import numpy as np

FeatureMap = Callable[..., np.array]
PPQ2Feature = namedtuple('PPQ2Feature', 'name', defaults=['pure_poly'])
RFQ2Feature = namedtuple('RFQ2Feature', 'lscale, name', defaults=[1, 'rbf_fourier'])

def pure_poli_features(x: np.array, order: int) -> np.array:
    """ Calculate pure polinomial features matrix for x. """
    return np.power(x[:, None], np.arange(order))

def gaussian_kernel_features(
        x: np.array, 
        order: int, 
        lscale: float = 1, 
        domain_bound: float = 1,
    ) -> np.array:
    """ 
    Calculate Gaussian (squared exp.) kernel features matrix for x. 

    References: "Hilbert Space Methods for Reduced-Rank 
        Gaussian Process Regression. Simo Särkkä. (formula 56)"
    """
    x = (x + domain_bound)
    w_scaled = np.pi * np.arange(1, order + 1) / (2 * domain_bound)
    sd = np.sqrt(2 * np.pi) * lscale * np.exp(-np.power(lscale * w_scaled, 2) / 2)
    return np.sqrt(sd / domain_bound) * np.sin(np.outer(x, w_scaled)) 

def q2_poli_features(x: np.array, q: int) -> np.array:
    """ Calculate special quantized pure polinomial features matrix for x. """
    return np.power(x[:, None], [0, 2**q])

def q2_fourier_features(
    x: np.array, 
    q: int, 
    m_order: int, 
    k_d: int, 
    lscale: float = 1
) -> np.array:
    """ Calculate special quantized Gaussian (squared exp.) kernel features matrix for x. """
    return np.hstack(
        (
            np.exp(-1j * np.pi * x * m_order / (k_d * lscale))[:, None], 
            np.exp(1j * np.pi * (-x * m_order / k_d + x*(2**q)) / lscale)[:, None]
        ),
    )
