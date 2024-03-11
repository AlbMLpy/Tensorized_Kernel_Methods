from typing import Optional
from functools import partial

from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from ..features import pure_poli_features, gaussian_kernel_features
from ..cp_krr import cp_krr, predict_score

class CPKRR(BaseEstimator, RegressorMixin):
    """ TODO """
    def __init__(
        self, 
        rank: int = 1, 
        feature_map: str = 'pure_poly', 
        m_order: int = 2,
        init_type: str = 'k_mtx',
        n_epoch: int = 1, 
        alpha: int = 1, 
        random_state: Optional[int] = None,
        lscale: float = 1,
        domain_bound: float = 1
    ):
        self.rank = rank
        self.feature_map = feature_map
        self.m_order = m_order
        self.init_type = init_type
        self.n_epoch = n_epoch
        self.alpha = alpha
        self.random_state = random_state
        self.lscale = lscale
        self.domain_bound = domain_bound
    
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
    
    def fit(self, X, y):
        """ TODO """
        X, y = check_X_y(X, y)
        self._feature_mapping = self._prepare_feature_mapping()
        self.weights_ = cp_krr(
            X, y, self.m_order, self._feature_mapping, 
            self.rank, self.init_type, self.n_epoch, self.alpha, self.random_state
        )
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """ TODO """
        X = check_array(X)
        check_is_fitted(self, 'is_fitted_')
        return predict_score(X, self.weights_, self._feature_mapping)
    
    def score(self, X, y):
        """ TODO """
        return r2_score(y, self.predict(X))
