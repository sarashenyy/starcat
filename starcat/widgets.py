import numpy as np


def round_to_step(arr, step):
    return np.round(arr / step) * step

