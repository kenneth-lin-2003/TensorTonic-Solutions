def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    # Write code here
    import numpy as np
    values = np.array(values)
    transitions = np.array(transitions)
    rewards = np.array(rewards)
    return np.max(rewards + gamma * np.dot(transitions, values), axis=1).tolist()