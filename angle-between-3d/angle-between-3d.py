import numpy as np

def angle_between_3d(v, w):
    """
    Compute the angle (in radians) between two 3D vectors.
    """
    if np.all(v == 0) or np.all(w == 0):
        return np.nan
    return np.arccos(np.clip(np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w)) , -1, 1))
    # Your code here
    pass