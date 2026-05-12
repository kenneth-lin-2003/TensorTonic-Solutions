import numpy as np

def detect_skew(train_dist, serving_dist, threshold=0.2, eps=1e-10):
    """
    Detect train-serving skew using PSI.
    """
    # Write code here
    ret = {}
    for k, v in train_dist.items():
        t = np.array(v, dtype=np.float64) + eps
        p = np.array(serving_dist[k], dtype=np.float64) + eps
        psi = np.sum((p - t) * np.log(p / t))
        ret[k] = {"psi":psi, "skewed":bool(psi>=threshold)}
    return ret
    pass