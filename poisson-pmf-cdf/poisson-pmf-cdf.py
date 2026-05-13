import numpy as np

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    # Write code here
    import math
    v_fac = np.vectorize(math.factorial)
    i = np.arange(k+1)
    e = np.exp(-lam)
    pmf = e * (lam ** k) / v_fac(k)
    cdf = np.sum(e * (lam ** i) / v_fac(i))
    return (pmf, cdf)
    pass