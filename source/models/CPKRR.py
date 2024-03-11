from typing import Optional
from functools import partial

from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from ..features import FeatureMap, pure_poli_features
from ..cp_krr import cp_krr, predict_score

DEFAULT_FD = 2
DEFAULT_FM = partial(pure_poli_features, order=DEFAULT_FD)

class CPKRR(BaseEstimator, RegressorMixin):
    """ TODO """
    def __init__(
        self, 
        rank: int = 1, 
        feature_map: FeatureMap = DEFAULT_FM, 
        m_order: int = DEFAULT_FD,
        init_type: str = 'k_mtx',
        n_epoch: int = 1, 
        alpha: int = 1, 
        random_state: Optional[int] = None,
    ):
        self.rank = rank
        self.feature_map = feature_map
        self.m_order = m_order
        self.init_type = init_type
        self.n_epoch = n_epoch
        self.alpha = alpha
        self.random_state = random_state
    
    def fit(self, X, y):
        """ TODO """
        X, y = check_X_y(X, y)
        self.weights_ = cp_krr(
            X, y, self.m_order, self.feature_map, 
            self.rank, self.init_type, self.n_epoch, self.alpha, self.random_state
        )
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """ TODO """
        X = check_array(X)
        check_is_fitted(self, 'is_fitted_')
        return predict_score(X, self.weights_, self.feature_map)
    
    def score(self, X, y):
        """ TODO """
        return r2_score(y, self.predict(X))
