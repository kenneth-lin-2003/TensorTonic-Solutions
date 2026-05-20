def hit_rate_at_k(recommendations, ground_truth, k):
    """
    Compute the hit rate at K.
    """
    # Write code here
    total, yes = len(ground_truth), 0
    for i in range(total):
        r, g = set(recommendations[i][:k]), set(ground_truth[i])
        tmp = r & g
        if len(tmp) != 0:
            yes += 1
    return yes / total