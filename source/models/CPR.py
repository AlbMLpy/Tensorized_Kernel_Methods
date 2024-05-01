from typing import Optional, Callable
from functools import partial

import numpy as np

from source.models.QCPR import QCPR
from source.features import pure_poli_features, gaussian_kernel_features

class CPR(QCPR):
    """ TODO """
    def __init__(
        self, 
        rank: int = 1, 
        feature_map: str = 'pure_poly', 
        m_order: int = 2,
        init_type: str = 'kj_vec',
        n_epoch: int = 1, 
        alpha: int = 1, 
        random_state: Optional[int] = None,
        lscale: float = 1,
        domain_bound: float = 1,
        callback: Optional[Callable] = None,
    ):
        super().__init__(
            rank, feature_map, m_order, init_type, 
            n_epoch, alpha, random_state, lscale, callback,
        )
        self.domain_bound = domain_bound
        self._quantized = False
        self._dtype = np.float64

    def _prepare_feature_mapping(self):
        if self.feature_map == 'pure_poly':
            return partial(pure_poli_features, order=self.m_order)
        elif self.feature_map == 'rbf_fourier':
            return partial(
                gaussian_kernel_features, 
                order=self.m_order, 
                lscale=self.lscale, 
                domain_bound=self.domain_bound
            )
        else:
            raise ValueError(f'Bad feature_map = "{self.feature_map}". See docs.')
