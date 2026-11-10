# Sales Prediction

A beginner machine learning project that predicts sales from advertising spending.

## Dataset

The dataset contains 200 observations and these columns:

- `TV` — TV advertising spend
- `radio` — radio advertising spend
- `newspaper` — newspaper advertising spend
- `sales` — value predicted by the model

The extra row-number column is removed by the script. This dataset does not contain a target-segment column.

Dataset source: [Advertising.csv](https://www.statlearning.com/s/Advertising.csv)

## What the project does

- Loads and inspects the data, handles missing feature values, and removes duplicate rows.
- Creates advertising-versus-sales and correlation charts.
- Trains and compares Linear Regression and Random Forest models.
- Evaluates the models with MAE, RMSE, and R².
- Predicts sales for TV, radio, and newspaper budgets entered by the user.
- Saves the charts, model comparison, and trained model in `outputs/`.

## Run the project

Install the required packages:

```bash
pip install pandas matplotlib scikit-learn joblib
```

Then run:

```bash
python main.py
```

When prompted, enter the TV, radio, and newspaper advertising budgets.

## Results from this run

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 0.66 | 0.79 | 0.98 |
| Linear Regression | 1.46 | 1.78 | 0.90 |

The Random Forest had the lower MAE in this train/test split. The correlations and model results describe patterns in this dataset; they do not prove that advertising caused sales changes.
