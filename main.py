from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Project folder aur files ke locations
PROJECT_FOLDER = Path(__file__).resolve().parent
CSV_FILE = PROJECT_FOLDER / "data" / "raw" / "Advertising.csv"
OUTPUT_FOLDER = PROJECT_FOLDER / "outputs"
OUTPUT_FOLDER.mkdir(exist_ok=True)


# 1. Dataset load karo
if not CSV_FILE.exists():
    raise FileNotFoundError(
        f"CSV file nahi mili: {CSV_FILE}\n"
        "Check karo ki Advertising.csv data/raw folder ke andar hai."
    )

data = pd.read_csv(CSV_FILE)

# Column names ke aas-paas extra spaces hatao
data.columns = data.columns.str.strip()

# CSV ka extra row-number column hatao
extra_columns = [
    column for column in data.columns
    if column.lower().startswith("unnamed:")
]
data = data.drop(columns=extra_columns)

# Is dataset mein ye chaar columns hone chahiye
required_columns = ["TV", "radio", "newspaper", "sales"]
missing_columns = [
    column for column in required_columns
    if column not in data.columns
]
if missing_columns:
    raise ValueError(f"CSV mein ye columns nahi mile: {missing_columns}")

# In columns ko numbers mein badlo; invalid values ko blank maana jayega
for column in required_columns:
    data[column] = pd.to_numeric(data[column], errors="coerce")

# Sales missing ho toh us row se model train nahi ho sakta
data = data.dropna(subset=["sales"])
data = data.drop_duplicates()

print("Dataset ki rows aur columns:", data.shape)
print("\nPehli 5 rows:")
print(data.head())

print("\nMissing values:")
print(data[required_columns].isna().sum())

print("\nSales ke saath feature correlations:")
print(data[required_columns].corr()["sales"].sort_values(ascending=False))


# 2. Advertising spend aur sales ke charts banao
channels = ["TV", "radio", "newspaper"]

figure, axes = plt.subplots(1, 3, figsize=(15, 4))

for axis, channel in zip(axes, channels):
    axis.scatter(data[channel], data["sales"], alpha=0.7)
    axis.set_title(f"{channel} spend vs sales")
    axis.set_xlabel(f"{channel} advertising spend")
    axis.set_ylabel("Sales")

figure.tight_layout()
figure.savefig(OUTPUT_FOLDER / "advertising_vs_sales.png")
plt.close(figure)


# Correlation heatmap
correlation = data[required_columns].corr()

figure, axis = plt.subplots(figsize=(6, 5))
image = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
axis.set_xticks(range(len(required_columns)))
axis.set_yticks(range(len(required_columns)))
axis.set_xticklabels(required_columns, rotation=45, ha="right")
axis.set_yticklabels(required_columns)
axis.set_title("Feature correlation")
figure.colorbar(image, ax=axis)

for row in range(len(required_columns)):
    for column in range(len(required_columns)):
        axis.text(
            column,
            row,
            f"{correlation.iloc[row, column]:.2f}",
            ha="center",
            va="center",
        )

figure.tight_layout()
figure.savefig(OUTPUT_FOLDER / "correlation_heatmap.png")
plt.close(figure)


# 3. Features aur target alag karo
# Features = advertising spend; target = sales
X = data[channels]
y = data["sales"]

# Data ka kuch hissa model ko test karne ke liye alag rakho
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


# 4. Do regression models train karke compare karo
models = {
    "Linear Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LinearRegression(),
    ),
    "Random Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            min_samples_leaf=2,
        ),
    ),
}

results = []
trained_models = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    trained_models[model_name] = model

    results.append(
        {
            "Model": model_name,
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
            "R2": r2_score(y_test, predictions),
        }
    )

# MAE jitna kam, test data par average prediction error utna kam
results_table = pd.DataFrame(results).sort_values("MAE")
results_table.to_csv(OUTPUT_FOLDER / "model_comparison.csv", index=False)

print("\nModel comparison:")
print(results_table.to_string(index=False))


# 5. MAE ke hisaab se behtar model chuno aur save karo
best_model_name = results_table.iloc[0]["Model"]
best_model = trained_models[best_model_name]
best_predictions = best_model.predict(X_test)

joblib.dump(best_model, OUTPUT_FOLDER / "best_sales_model.joblib")

print(f"\nSelected model: {best_model_name}")
print("Model aur charts 'outputs' folder mein save ho gaye hain.")


# Actual sales aur predicted sales ka chart
figure, axis = plt.subplots(figsize=(6, 5))
axis.scatter(y_test, best_predictions, alpha=0.7)

lowest = min(y_test.min(), best_predictions.min())
highest = max(y_test.max(), best_predictions.max())
axis.plot([lowest, highest], [lowest, highest], "r--")

axis.set_title(f"Actual vs predicted sales: {best_model_name}")
axis.set_xlabel("Actual sales")
axis.set_ylabel("Predicted sales")
figure.tight_layout()
figure.savefig(OUTPUT_FOLDER / "actual_vs_predicted.png")
plt.close(figure)

print("\nNew campaign ke ad budgets enter karo:")

tv_spend = float(input("TV budget: "))
radio_spend = float(input("Radio budget: "))
newspaper_spend = float(input("Newspaper budget: "))

new_campaign = pd.DataFrame(
    [{
        "TV": tv_spend,
        "radio": radio_spend,
        "newspaper": newspaper_spend,
    }]
)

predicted_sales = best_model.predict(new_campaign)[0]
print(f"Predicted sales: {predicted_sales:.2f}")