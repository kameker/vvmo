import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from iris_data import variant6_x, variant6_y


class GaussianNaiveBayes:
    def fit(self, x, y):
        self.classes = np.unique(y)
        self.means = {}
        self.variances = {}
        self.priors = {}

        for class_label in self.classes:
            class_objects = x[y == class_label]
            self.means[class_label] = np.mean(class_objects, axis=0)
            self.variances[class_label] = np.var(class_objects, axis=0) + 1e-9
            self.priors[class_label] = len(class_objects) / len(x)

    def predict_one(self, obj):
        scores = {}

        for class_label in self.classes:
            mean = self.means[class_label]
            variance = self.variances[class_label]

            log_prior = np.log(self.priors[class_label])
            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * variance) + (obj - mean) ** 2 / variance
            )
            scores[class_label] = log_prior + log_likelihood

        return max(scores, key=scores.get)

    def predict(self, x):
        return np.array([self.predict_one(obj) for obj in x])


def draw_sample(ax, x, y, title):
    styles = {
        -1: ("royalblue", "o"),
        1: ("crimson", "^"),
    }

    for class_label in [-1, 1]:
        objects = x[y == class_label]
        color, marker = styles[class_label]
        ax.scatter(
            objects[:, 0],
            objects[:, 1],
            color=color,
            marker=marker,
            s=55,
            label=f"Класс {class_label}",
            edgecolors="black",
            linewidths=0.5,
        )

    ax.set_title(title)
    ax.set_xlabel("Признак 1")
    ax.set_ylabel("Признак 2")
    ax.grid(alpha=0.3)
    ax.legend()


def main():
    x = np.array(variant6_x)
    y = np.array(variant6_y)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = GaussianNaiveBayes()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = np.mean(predictions == y_test)

    print("Истинные классы:      ", y_test)
    print("Предсказанные классы: ", predictions)
    print(f"Точность: {accuracy:.2%}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    draw_sample(axes[0], x_train, y_train, "Обучающая выборка")
    draw_sample(axes[1], x_test, y_test, "Тестовая выборка")
    fig.suptitle("Наивный байесовский классификатор — вариант 6")
    fig.tight_layout()
    fig.savefig("naive_bayes_variant6.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
