import matplotlib.pyplot as plt


def plotData(data):
    """Визуализирует численность населения и прибыль."""
    plt.figure(figsize=(8, 6))
    plt.plot(
        data[:, 0],
        data[:, 1],
        "rx",
        markersize=8,
        label="Обучающие данные",
    )
    plt.xlabel("Численность населения, десятки тысяч человек")
    plt.ylabel("Прибыль, десятки тысяч долларов")
    plt.grid(alpha=0.3)
