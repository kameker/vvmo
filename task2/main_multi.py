"""Часть 2: линейная регрессия с несколькими переменными."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from featureNormalize import featureNormalize
from gradientDescent import gradientDescent
from normalEqn import normalEqn


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "example" / "data2.txt"


def main():
    data = np.loadtxt(DATA_FILE, delimiter=",")
    X_original = data[:, :2]
    y = data[:, 2:3]
    m = data.shape[0]

    X_normalized, mu, sigma = featureNormalize(X_original)
    X = np.column_stack((np.ones(m), X_normalized))

    learning_rates = [0.001, 0.003, 0.01, 0.03, 0.1]
    histories = {}
    plt.figure(figsize=(9, 6))

    for alpha in learning_rates:
        _, history = gradientDescent(X, y, np.zeros((3, 1)), alpha, 400)
        histories[alpha] = history
        plt.plot(history, label=f"alpha = {alpha}")

    plt.xlabel("Число итераций")
    plt.ylabel("Значение стоимостной функции")
    plt.title("Влияние скорости обучения на сходимость")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "multi_variable_convergence.png", dpi=150)
    plt.close()

    theta_gradient, _ = gradientDescent(
        X, y, np.zeros((3, 1)), alpha=0.01, num_iters=400
    )
    house_normalized = np.array(
        [[1, (1650 - mu[0]) / sigma[0], (3 - mu[1]) / sigma[1]]]
    )
    prediction_gradient = float(np.dot(house_normalized, theta_gradient)[0, 0])

    X_equation = np.column_stack((np.ones(m), X_original))
    theta_equation = normalEqn(X_equation, y)
    prediction_equation = float(np.dot([[1, 1650, 3]], theta_equation)[0, 0])

    print("Линейная регрессия с несколькими переменными")
    print(f"Средние значения признаков: {mu}")
    print(f"Среднеквадратические отклонения: {sigma}")
    print("\nКонечная стоимость для разных значений alpha:")
    for alpha, history in histories.items():
        print(f"  alpha = {alpha:>5}: J = {history[-1]:.2f}")
    print("\nПараметры после градиентного спуска:")
    print(theta_gradient)
    print("Параметры по нормальному уравнению:")
    print(theta_equation)
    print(f"\nПрогноз методом градиентного спуска: ${prediction_gradient:.2f}")
    print(f"Прогноз по нормальному уравнению:   ${prediction_equation:.2f}")
    print(f"Разница прогнозов:                  ${abs(prediction_gradient - prediction_equation):.2f}")


if __name__ == "__main__":
    main()
