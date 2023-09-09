import logging
import time
from functools import wraps

import numpy as np


def round_to_step(arr, step):
    return np.round(arr / step) * step


def log_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        run_time = end_time - start_time
        logging.info(f"time init() : {run_time:.4f} s")
        return result

    return wrapper
