import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.array(x)
    if rng is None:
        mask = np.random.random(x.shape)
    else:
        mask = rng.random(x.shape)
    mask = np.where(mask < p, 0, 1 / (1 - p))
    return (x * mask, mask)
    pass