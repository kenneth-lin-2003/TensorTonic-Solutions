import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    y = y[:,np.newaxis]
    N = X.shape[0]
    w = np.zeros(X.shape[1])[:,np.newaxis]
    b = 0
    # Write code here
    for _ in range(steps):
        err = _sigmoid(X @ w + b) - y
        w = w - lr * (X.T @ err) / N
        b = b - lr * np.mean(err)
    w = w.reshape(-1)
    return w, b