import time 
from typing import Optional

import numpy as np
from memory_profiler import memory_usage

from .loss import mse_l2_loss

def prepare_callback_mse_wl2():
    def callback_function(
        y: np.ndarray, 
        y_pred: np.ndarray, 
        weights: np.ndarray,
        *args,
        **kwargs,
    ):
        if not hasattr(callback_function, 'data'):
            callback_function.data = [] 
        if 'alpha' not in kwargs:
            raise ValueError("Alpha parameter for L2 regularization must be provided.")
        
        value = mse_l2_loss(y, y_pred, weights, kwargs['alpha'])
        callback_function.data.append(value)
    return callback_function

def performance_decorator(enabled=True, buffer: Optional[dict] = None):
    def actual_decorator(func):
        def wrapper(*args, **kwargs):
            if enabled:
                mem_usage_before = memory_usage(max_usage=True)
                s_time = time.time()
                result = func(*args, **kwargs)
                elapsed_time = time.time() - s_time
                mem_usage_after = memory_usage(max_usage=True)
                memory_used = mem_usage_after - mem_usage_before
                if buffer is not None:
                    buffer[func.__name__ + '_time'] = elapsed_time
                    buffer[func.__name__ + '_mem'] = memory_used
                else:
                    print(f"{func.__name__} used {memory_used} MiB and took {elapsed_time} seconds")
            else:
                result = func(*args, **kwargs)
            return result
        return wrapper
    return actual_decorator
