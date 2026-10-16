"""Multivariable linear regression for weekly IBM data."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "task1" / "IBM.csv", sep=";")
df["DATE"] = pd.to_datetime(df["<DATE>"], format="%d.%m.%y")
df = df.rename(columns={"<OPEN>": "OPEN", "<HIGH>": "HIGH", "<LOW>": "LOW", "<CLOSE>": "CLOSE", "<VOL>": "VOL"})

features = ["OPEN", "HIGH", "LOW", "VOL"]
split = int(len(df) * 0.8)
X_train, X_test = df.loc[:split - 1, features], df.loc[split:, features]
y_train, y_test = df.loc[:split - 1, "CLOSE"], df.loc[split:, "CLOSE"]

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

pd.Series({"MAE": mean_absolute_error(y_test, y_pred), "RMSE": mean_squared_error(y_test, y_pred) ** 0.5, "R2": r2_score(y_test, y_pred)}).to_csv(OUT / "ibm_regression_metrics.csv", header=["value"])
pd.DataFrame({"feature": features, "coefficient": model.coef_}).to_csv(OUT / "ibm_regression_coefficients.csv", index=False)

result = df.loc[split:, ["DATE", "CLOSE"]].copy()
result["PREDICTED"] = y_pred
ax = result.plot(x="DATE", y=["CLOSE", "PREDICTED"], figsize=(10, 5), title="IBM: actual and predicted closing price")
ax.set_ylabel("Price")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "ibm_regression_predictions.png", dpi=160)
