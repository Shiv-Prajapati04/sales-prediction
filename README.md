# Sales Prediction

A beginner machine learning project that predicts sales from advertising spending.

## Dataset

The dataset has 200 rows and these columns:

- `TV`
- `radio`
- `newspaper`
- `sales` — the value this project predicts

Dataset source: [Advertising.csv](https://www.statlearning.com/s/Advertising.csv)

This dataset does not include a target-segment column.

## Models

The project compares Linear Regression and Random Forest. On this run, Random Forest had the lower MAE.

## Run the project

Install the required packages:

```bash
pip install pandas matplotlib scikit-learn joblib
