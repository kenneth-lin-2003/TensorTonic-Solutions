import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    # Write code here
    var = np.sum(np.square(x - np.mean(x))) / (len(x) - 1)
    std = np.sqrt(var)
    return (var, std)
    pass