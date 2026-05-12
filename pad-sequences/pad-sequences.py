import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if not seqs:
        return np.array([[]])
    if max_len is None:
        max_len = max([len(s) for s in seqs])
    N = len(seqs)
    ret = np.full((N, max_len), pad_value)
    for i in range(N):
        s = np.array(seqs[i])
        l = min(max_len, s.size)
        ret[i, :l] = s[:l]
    return ret
    pass