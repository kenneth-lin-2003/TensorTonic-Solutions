import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    """
    Compute confusion matrix with optional normalization.
    """
    # Write code here
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_true.size == 0 or y_pred.size == 0:
        if num_classes is not None:
            return np.zeros((num_classes, num_classes))
        return np.array([])
    if num_classes is None:
        num_classes = max(np.max(y_true), np.max(y_pred))+1
    if np.any((y_true >= num_classes) | (y_true < 0)) or np.any((y_pred >= num_classes) | (y_pred < 0)):
        if num_classes is not None:
            return np.zeros((num_classes, num_classes))
        return np.array([])
    mat = np.bincount(y_true * num_classes + y_pred, minlength=num_classes*num_classes).reshape(num_classes, num_classes)
    epi = 1e-7
    if normalize == 'true':
        mat = mat / (np.sum(mat, axis=1, keepdims=True)+epi)
    elif normalize == 'pred':
        mat = mat / (np.sum(mat, axis=0, keepdims=True)+epi)
    elif normalize == 'all':
        mat = mat / (np.sum(mat)+epi)
    return mat
    pass