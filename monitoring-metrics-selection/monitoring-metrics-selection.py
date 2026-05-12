def compute_monitoring_metrics(system_type, y_true, y_pred):
    """
    Compute the appropriate monitoring metrics for the given system type.
    """
    # Write code here
    import numpy as np
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    if system_type == "classification":
        mask = (y_true == y_pred)
        accuracy = np.sum(mask) / y_pred.size
        p, t =  np.sum(y_pred), np.sum(y_true)
        if p == 0:
            precision = 0
        else:
            precision = np.sum(y_true & mask) / p
        if t == 0:
            recall = 0
        else:
            recall = np.sum(y_true & mask) / t
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = (2 * precision * recall) / (precision + recall)
        ret = [("accuracy", accuracy), ("f1", f1), ("precision", precision), ("recall", recall)]
    elif system_type == "regression":
        dif = y_pred - y_true
        mae = np.mean(np.abs(dif))
        rmse = np.sqrt(np.mean(np.square(dif)))
        ret = [("mae", mae), ("rmse", rmse)]
    elif system_type == "ranking":
        idx = np.argsort(y_pred)[::-1][:3]
        top3 = np.sum(y_true[idx])
        t = np.sum(y_true)
        precision_at_3 = top3 / 3
        if t == 0:
            recall_at_3 = 0
        else:
            recall_at_3 = top3 / t
        ret = [("precision_at_3", precision_at_3), ("recall_at_3", recall_at_3)]
    return ret
    pass