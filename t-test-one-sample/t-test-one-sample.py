import numpy as np

def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """
    x = np.array(x)
    mean = np.mean(x)
    s = np.std(x, ddof=1)
    n = len(x)
    return (mean - mu0) / (s / np.sqrt(n))