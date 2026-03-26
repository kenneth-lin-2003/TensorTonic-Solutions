import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    # Write code here
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    if X_train.ndim == 1:
        X_train = X_train[:, np.newaxis]
    if X_test.ndim == 1:
        X_test = X_test[:, np.newaxis]
    _X_train = np.sum(np.square(X_train), axis=-1, keepdims=True)
    _X_test = np.sum(np.square(X_test), axis=-1, keepdims=True)
    _X_train = _X_train.transpose(1, 0)
    Q = _X_test + _X_train - 2 * (X_test @ X_train.T)
    idx = np.argsort(Q, axis = -1)
    if k > X_train.shape[0]:
        ret = np.full((X_test.shape[0], k), -1)
        ret[:, :X_train.shape[0]] = idx
        return ret
    return idx[:,:k]
    pass