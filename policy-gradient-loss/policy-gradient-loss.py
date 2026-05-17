def policy_gradient_loss(log_probs, rewards, gamma):
    import numpy as np

    log_probs = np.asarray(log_probs, dtype=float)
    rewards = np.asarray(rewards, dtype=float)

    if gamma == 0:
        return float(-np.mean(log_probs * (rewards - np.mean(rewards))))

    T = len(rewards)

    discounts = gamma ** np.arange(T)

    returns = np.cumsum((rewards * discounts)[::-1])[::-1] / discounts

    advantages = returns - returns.mean()

    loss = -np.mean(log_probs * advantages)

    return float(loss)