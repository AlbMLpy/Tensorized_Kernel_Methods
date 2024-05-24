from typing import Callable
from collections import namedtuple

import numpy as np

FeatureMap = Callable[..., np.ndarray]
PPQ2Feature = namedtuple('PPQ2Feature', 'name', defaults=['pure_poly'])
RFQ2Feature = namedtuple('RFQ2Feature', 'lscale, name', defaults=[1, 'rbf_fourier'])

def pure_poli_features(
    x: np.ndarray, 
    q: int, # Dummy
    order: int
) -> np.ndarray:
    """ 
    Pure polinomial features matrix for x. 

    References: "Quantized Fourier and Polynomial Features for more 
        Expressive Tensor Network Models", Frederiek Wesel, Kim Batselier, (Definition 3.1).
    """
    return np.power(x[:, None], np.arange(order))

def gaussian_kernel_features(
    x: np.ndarray,
    q: int,  # Dummy
    order: int, 
    lscale: float = 1, 
    domain_bound: float = 1,
) -> np.ndarray:
    """ 
    Gaussian (squared exp.) kernel features matrix for x. 

    References: "Hilbert Space Methods for Reduced-Rank Gaussian Process Regression", 
        Simo Särkkä, (formulas 56, 68(d=1, s=1)).
    """
    x = (x + domain_bound)
    w_scaled = np.pi * np.arange(1, order + 1) / (2 * domain_bound)
    sd = np.sqrt(2 * np.pi) * lscale * np.exp(-np.power(lscale * w_scaled, 2) / 2)
    return np.sqrt(sd / domain_bound) * np.sin(np.outer(x, w_scaled)) 

def q2_poli_features(x: np.ndarray, q: int) -> np.ndarray:
    """ 
    Quantized pure polinomial features matrix for x. 
    
    References: "Quantized Fourier and Polynomial Features for more 
        Expressive Tensor Network Models", Frederiek Wesel, Kim Batselier, (Definition 3.4).
    """
    return np.power(x[:, None], [0, 2**q])

def q2_fourier_features(
    x: np.ndarray, 
    q: int, 
    m_order: int, 
    k_d: int, 
    p_scale: float = 1
) -> np.ndarray:
    """ 
    Fourier Features matrix for x. 

    References: 
        - "Learning multidimensional Fourier series with tensor trains",
            Sander Wahls, Visa Koivunen, H Vincent Poor, Michel Verhaegen.
        - "Quantized Fourier and Polynomial Features for more 
            Expressive Tensor Network Models", Frederiek Wesel, Kim Batselier, (Corollary 3.6).
    """
    return np.hstack(
        (
            np.exp(-1j * np.pi * x * m_order / (k_d * p_scale))[:, None], 
            np.exp(1j * np.pi * (-x * m_order / k_d + x*(2**q)) / p_scale)[:, None]
        ),
    )
