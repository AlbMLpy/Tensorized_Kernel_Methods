from typing import Optional, Callable
from functools import partial

import numpy as np
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from ..features import PPQ2Feature, q2_poli_features, q2_fourier_features
from ..q_cpr_f import q_cpr_f, predict_score

class QCPRf(BaseEstimator, RegressorMixin):
    """ Quantized CP Regression Model with feature relevance learning (QCPRf). """
    def __init__(
        self, 
        rank: int = 1, 
        fmaps_list: list[tuple] = [PPQ2Feature(),],
        m_order: int = 2,
        init_type: str = 'kj_vec',
        n_epoch: int = 1, 
        alpha: int = 1, 
        beta: int = 1,
        random_state: Optional[int] = None,
        callback: Optional[Callable] = None,
    ):
        _check_input_params(m_order)  
        self.rank = rank
        self.fmaps_list = fmaps_list
        self.m_order = m_order
        self.init_type = init_type
        self.n_epoch = n_epoch
        self.alpha = alpha
        self.beta = beta
        self.random_state = random_state
        self.callback = callback
        self._dtype = None
    
    def _prepare_feature_mappings(self):
        mappings = []
        for feature in self.fmaps_list:
            if feature.name == 'pure_poly':
                mappings.append(q2_poli_features)
            elif feature.name == 'rbf_fourier':
                mappings.append(
                    partial(
                        q2_fourier_features, 
                        m_order=self.m_order, 
                        k_d=int(np.log2(self.m_order)), 
                        lscale=feature.lscale,
                    )
                )
                self._dtype = np.complex128
            else:
                raise ValueError(f'Bad feature_map name = "{feature.name}". See docs.')
        self._dtype = np.float64 if self._dtype is None else self._dtype
        return mappings

    def fit(self, X, y):
        """ TODO """
        X, y = check_X_y(X, y)
        self._feature_maps_list = self._prepare_feature_mappings()
        self.weights_, self.lambdas_, self.kd_ = q_cpr_f(
            X, y, self.m_order, self._feature_maps_list, 
            self.rank, self.init_type, self.n_epoch, self.alpha, 
            self.beta, self.random_state, self._dtype, self.callback
        )
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """ TODO """
        X = check_array(X)
        check_is_fitted(self, 'is_fitted_')
        return predict_score(
            X, self.kd_, self.weights_, self.lambdas_, self._feature_maps_list)
    
    def score(self, X, y):
        """ TODO """
        return r2_score(y, self.predict(X))
    
def _check_input_params(m_order: int) -> None:
    if m_order & (m_order - 1):
        raise ValueError(f"m_order should be a power of 2, but it is {m_order}.")
