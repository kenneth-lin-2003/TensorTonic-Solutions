def bilinear_resize(image: list, new_h: int, new_w: int) -> list:
    """
    Returns a two-dimensional list with shape (new_h, new_w).
    """
    import numpy as np
    image = np.array(image)
    h, w = image.shape
    y = np.zeros(new_h) if new_h == 1 else np.arange(new_h) * (h-1) / (new_h-1)
    x = np.zeros(new_w) if new_w == 1 else np.arange(new_w) * (w-1) / (new_w-1)
    ly = np.floor(y).astype(int)[:,None]
    lx = np.floor(x).astype(int)[None,:]
    ry = np.clip(ly+1, a_min=None, a_max=h-1)
    rx = np.clip(lx+1, a_min=None, a_max=w-1)
    dx = (x[None,:] - lx)
    dy = (y[:,None] - ly)
    ret = (image[ly, lx] * (1 - dx) + image[ly, rx] * dx) * (1 - dy) + (image[ry, lx] * (1 - dx) + image[ry, rx] * dx) * dy
    return ret.tolist()
    