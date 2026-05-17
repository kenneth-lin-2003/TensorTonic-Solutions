def policy_gradient_loss(log_probs, rewards, gamma):
    """
    Compute REINFORCE policy gradient loss with mean-return baseline.
    """
    # Write code here
    import numpy as np
    log_probs = np.array(log_probs)
    rewards = np.array(rewards)
    T = rewards.shape[0]
    A = np.zeros(T)
    A[T-1] = rewards[T-1]
    for t in range(T-2, -1, -1):
        A[t] = rewards[t] + gamma * A[t+1]
    A = A - np.mean(A)
    ans = -np.mean(log_probs * A)
    return float(ans)