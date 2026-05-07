def novelty_score(recommendations, item_counts, n_users):
    """
    Compute the average novelty of a recommendation list.
    """
    # Write code here
    import numpy as np
    recommendations = np.array(recommendations)
    item_counts = np.array(item_counts)
    if recommendations.size == 0:
        return 0.0
    return np.mean(-np.log2(item_counts[recommendations] / n_users))
    