import numpy as np

def tfidf_vectorizer(documents):
    tokenized = [d.split() for d in documents]
    vocabulary = sorted(set(w for doc in tokenized for w in doc))
    stoi = {w: i for i, w in enumerate(vocabulary)}

    rows = []
    cols = []

    for i, words in enumerate(tokenized):
        for w in words:
            rows.append(i)
            cols.append(stoi[w])

    N = len(documents)
    V = len(vocabulary)

    tf_counts = np.zeros((N, V), dtype=float)
    np.add.at(tf_counts, (rows, cols), 1)

    doc_lengths = np.array([len(words) for words in tokenized])[:, None]
    tf = tf_counts / doc_lengths

    df = (tf_counts > 0).sum(axis=0)
    idf = np.where(df > 0, np.log(N / df), 0)

    tfidf = tf * idf

    return tfidf, vocabulary