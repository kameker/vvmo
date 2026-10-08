import numpy as np


def normalEqn(X, y):
    """Вычисляет параметры линейной регрессии нормальным уравнением."""
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
