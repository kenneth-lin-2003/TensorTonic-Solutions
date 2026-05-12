def detect_drift(reference_counts, production_counts, threshold):
    """
    Compare reference and production distributions to detect data drift.
    """
    # Write code here
    import numpy as np
    reference_counts = np.array(reference_counts)
    production_counts = np.array(production_counts)
    reference_counts = reference_counts / np.sum(reference_counts)
    production_counts = production_counts / np.sum(production_counts)
    tvd = 0.5 * np.sum(np.abs(reference_counts - production_counts))
    return {"score":tvd, "drift_detected":bool(tvd > threshold)}
    pass