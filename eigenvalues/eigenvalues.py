import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    try:
        # 將輸入轉換為 NumPy 陣列（如果形狀不對，這行會直接報錯並跳到 except）
        matrix = np.asarray(matrix, dtype=float)
    except (ValueError, TypeError):
        return None

    # 確保是二維且是方陣
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    eig = np.linalg.eigvals(matrix)
    idx = np.lexsort((eig.imag, eig.real))
    return eig[idx]