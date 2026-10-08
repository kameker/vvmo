from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_FILE = Path(__file__).with_name("IBM.csv")
CHART_FILE = Path(__file__).with_name("ibm_monthly_volume.png")

MONTH_NAMES_RU = {
    1: "Январь",
    2: "Февраль",
    3: "Март",
    4: "Апрель",
    5: "Май",
    6: "Июнь",
    7: "Июль",
    8: "Август",
    9: "Сентябрь",
    10: "Октябрь",
    11: "Ноябрь",
    12: "Декабрь",
}


def load_data(path):
    data = pd.read_csv(path, sep=";")
    data["<DATE>"] = pd.to_datetime(data["<DATE>"], format="%d.%m.%y")
    return data.rename(
        columns={
            "<DATE>": "date",
            "<OPEN>": "open",
            "<HIGH>": "high",
            "<LOW>": "low",
            "<CLOSE>": "close",
            "<VOL>": "volume",
        }
    )


def calculate_statistics(data):
    monthly_volume = (
        data.assign(month_number=data["date"].dt.month)
        .groupby("month_number", as_index=False)["volume"]
        .sum()
        .sort_values("month_number")
    )
    monthly_volume["month"] = monthly_volume["month_number"].map(MONTH_NAMES_RU)
    monthly_volume = monthly_volume[["month_number", "month", "volume"]]

    yearly_close = (
        data.assign(year=data["date"].dt.year)
        .groupby("year", as_index=False)["close"]
        .mean()
        .rename(columns={"close": "average_close"})
    )
    yearly_close["average_close"] = yearly_close["average_close"].round(2)

    return monthly_volume, yearly_close


def plot_monthly_volume(monthly_volume, output_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(monthly_volume["month"], monthly_volume["volume"], color="steelblue")
    ax.set_title("Суммарный объём сделок с акциями IBM по месяцам")
    ax.set_xlabel("Месяц")
    ax.set_ylabel("Суммарный объём сделок")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    data = load_data(DATA_FILE)
    monthly_volume, yearly_close = calculate_statistics(data)

    print("Суммарный объём сделок по месяцам:")
    print(monthly_volume[["month", "volume"]].to_string(index=False))

    busiest = monthly_volume.loc[monthly_volume["volume"].idxmax()]
    print(
        f"\nНаибольший объём был в месяце: {busiest['month']} "
        f"({busiest['volume']:,.0f})."
    )

    print("\nСредняя цена акций на закрытии торгов по годам:")
    print(yearly_close.to_string(index=False))

    plot_monthly_volume(monthly_volume, CHART_FILE)
    print(f"\nДиаграмма сохранена в файл: {CHART_FILE.name}")


if __name__ == "__main__":
    main()
