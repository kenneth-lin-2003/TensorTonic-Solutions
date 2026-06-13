import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    # Write code here
    X_train, X_test = np.array(X_train), np.array(X_test)
    if X_test.ndim == 1:
        x = np.abs(X_test[:,None] - X_train[None,:])
        x = np.argsort(x, axis=-1)[:,:k]
        pad_len = max(0, k-x.shape[1])
        x = np.pad(x, ((0,0),(0,pad_len)), mode="constant", constant_values=-1)
        return x
    dtrain, dtest = X_train.shape[0], X_test.shape[0]
    x = X_test.reshape(dtest, 1, -1) - X_train.reshape(1, dtrain, -1)
    x = np.sum(np.square(x), axis=-1)
    x = np.argsort(x, axis=-1)[:,:k]
    pad_len = max(0, k-x.shape[1])
    x = np.pad(x, ((0,0),(0,pad_len)), mode="constant", constant_values=-1)
    return x