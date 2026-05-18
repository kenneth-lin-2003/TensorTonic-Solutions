import math

def roi_pool(feature_map, rois, output_size):
    """
    Apply ROI Pooling to extract fixed-size features.
    """
    # Write code here
    import numpy as np
    feature_map = np.array(feature_map)
    n, m = feature_map.shape
    ans = []
    for x1, y1, x2, y2 in rois:
        arr = feature_map[y1 : y2 + 1, x1 : x2 + 1]
        h = y2 - y1
        w = x2 - x1
        ret = [[0] * output_size for _ in range(output_size)]
        for i in range(output_size):
            for j in range(output_size):
                hs, he, ws, we = y1 + ((i * h) // output_size), y1 + (((i+1) * h) // output_size), x1 + ((j * w) // output_size), x1 + (((j+1) * w) // output_size)
                if hs == he:
                    he = hs + 1
                if ws == we:
                    we = ws + 1
                he = min(he, n)
                we = min(we, m)
                ret[i][j] = np.max(feature_map[hs:he, ws:we]).item()
        ans.append(ret)
    return ans