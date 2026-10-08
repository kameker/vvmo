import numpy as np

from computeCost import computeCost


def gradientDescent(X, y, theta, alpha, num_iters):
    """Находит параметры линейной регрессии методом градиентного спуска."""
    m = y.shape[0]
    J_history = []

    for _ in range(num_iters):
        errors = np.dot(X, theta) - y
        theta = theta - alpha / m * np.dot(X.T, errors)
        J_history.append(computeCost(X, y, theta))

    return theta, J_history
