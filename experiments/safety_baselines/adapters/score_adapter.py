import numpy as np

def score_from_x0(x, x0, alpha_bar):
    a = np.asarray(alpha_bar)
    if np.any(~np.isfinite(a)) or np.any((a <= 0) | (a >= 1)):
        raise ValueError('alpha_bar must lie strictly between zero and one')
    x, x0 = np.asarray(x), np.asarray(x0)
    if x.shape != x0.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(x0)):
        raise ValueError('finite, matching states required')
    return -(x - np.sqrt(a) * x0) / (1-a)
