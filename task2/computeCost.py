import numpy as np


def computeCost(X, y, theta):
    """Вычисляет среднеквадратичную стоимостную функцию линейной регрессии."""
    m = y.shape[0]
    errors = np.dot(X, theta) - y
    return float(np.sum(errors ** 2) / (2 * m))
