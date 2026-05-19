import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x = np.array(x)
    p = np.array(p)
    eps = 1e-6
    if np.abs(np.sum(p) - 1) > 1e-6:
        raise ValueError
    if len(x) != len(p):
        raise ValueError
    return float(np.sum(x * p))
    pass
