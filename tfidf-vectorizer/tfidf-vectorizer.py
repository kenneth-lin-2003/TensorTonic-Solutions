import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    vocabulary = sorted(list(set(w for d in documents for w in d.split())))
    stoi = {s:i for i,s in enumerate(vocabulary)}
    tf, idf, N = [], np.zeros(len(vocabulary)), len(documents)
    for d in documents:
        cnt = Counter(d.split())
        total = len(d.split())
        tf.append([cnt[v] / total for v in vocabulary])
        for k in cnt.keys():
            idf[stoi[k]] += 1
    tf = np.array(tf)
    idf = np.where(idf > 0, np.log(N / idf), 0)
    tfidf = tf * idf[None,:]
    return (tfidf, vocabulary)