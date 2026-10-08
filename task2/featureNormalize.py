import numpy as np


def featureNormalize(X):
    """Нормализует каждый признак на среднее и выборочное отклонение."""
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=1)
    X_norm = np.divide(X - mu, sigma)
    return X_norm, mu, sigma
