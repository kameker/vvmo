"""Часть 1: линейная регрессия с одной переменной."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from computeCost import computeCost
from gradientDescent import gradientDescent
from plotData import plotData


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "example" / "data1.txt"


def main():
    data = np.loadtxt(DATA_FILE, delimiter=",")
    m = data.shape[0]
    X = np.column_stack((np.ones(m), data[:, 0]))
    y = data[:, 1:2]

    plotData(data)
    plt.title("Исходные данные")
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "one_variable_data.png", dpi=150)
    plt.close()

    theta = np.zeros((2, 1))
    initial_cost = computeCost(X, y, theta)
    theta, cost_history = gradientDescent(X, y, theta, 0.01, 1500)

    plotData(data)
    plt.plot(X[:, 1], np.dot(X, theta), "b-", label="Линейная регрессия")
    plt.title("Линейная регрессия с одной переменной")
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "one_variable_regression.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(cost_history) + 1), cost_history)
    plt.xlabel("Число итераций")
    plt.ylabel("Значение стоимостной функции")
    plt.title("Сходимость градиентного спуска")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "one_variable_convergence.png", dpi=150)
    plt.close()

    predict_35000 = float(np.dot([[1, 3.5]], theta)[0, 0] * 10000)
    predict_70000 = float(np.dot([[1, 7.0]], theta)[0, 0] * 10000)

    print("Линейная регрессия с одной переменной")
    print(f"Начальное значение стоимостной функции: {initial_cost:.4f}")
    print(f"Параметры модели: {theta[0, 0]:.4f}, {theta[1, 0]:.4f}")
    print(f"Прогноз прибыли для населения 35 000: ${predict_35000:.2f}")
    print(f"Прогноз прибыли для населения 70 000: ${predict_70000:.2f}")


if __name__ == "__main__":
    main()
