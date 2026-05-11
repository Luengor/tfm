import numpy as np


def find_elbow(y: np.ndarray) -> int:
    """
    Finds the elbow point in a curve using the perpendicular distance method.
    Returns the index of the elbow point.
    """
    n = len(y)
    if n < 3:
        return 0
    x = np.arange(n)

    p1 = np.array([0, y[0]])
    p2 = np.array([n - 1, y[-1]])

    line_vec = p2 - p1
    norm = np.sqrt(np.sum(line_vec**2))
    if norm == 0:
        return 0

    line_vec_norm = line_vec / norm

    p1_to_p = np.column_stack((x, y)) - p1
    proj = np.outer(np.dot(p1_to_p, line_vec_norm), line_vec_norm)
    dist_to_line = np.sqrt(np.sum((p1_to_p - proj)**2, axis=1))

    return int(np.argmax(dist_to_line))
