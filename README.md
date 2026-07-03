# House Price Prediction — ML Regression Project

## Overview
This project predicts whether a house falls in a **Low**, **Medium**, or **High** price category using machine learning classification techniques. The dataset used is the Ames Housing Dataset which contains information about houses sold in Ames, Iowa.

Instead of predicting the exact price (regression), we convert the problem into a classification task by dividing houses into 3 price buckets — which makes it easier to evaluate and understand.

---

## Dataset
- **File:** `train.csv`
- **Source:** Ames Housing Dataset (Kaggle)
- **Rows:** 1460 houses
- **Columns:** 81 features (area, quality, year built, neighborhood, etc.)
- **Target Column:** `SalePrice` → converted to `PriceCategory` (Low / Medium / High)

---

## Problem Statement
> Given various features of a house (size, quality, location, age, etc.), can we predict whether it will be a low, medium, or high priced house?

---

## Price Categories
Sale prices are divided into 3 classes using percentiles:

| Class  | Price Range         |
|--------|---------------------|
| Low    | Below ~$139,000     |
| Medium | $139,000 – $189,000 |
| High   | Above ~$189,000     |

Each class has roughly equal number of houses (~480 each) which keeps the dataset balanced.

---

## Project Steps

1. **Import Libraries** — pandas, numpy, matplotlib, seaborn, scikit-learn
2. **Load Dataset** — read train.csv and explore basic info
3. **Create Target Variable** — convert SalePrice into Low/Medium/High labels
4. **EDA (Exploratory Data Analysis)** — understand patterns in the data through plots
5. **Preprocessing** — handle missing values, encode categorical columns, engineer new features
6. **Train/Test Split** — 80% training, 20% testing
7. **Train Models** — Logistic Regression, Decision Tree, Random Forest
8. **Evaluate Models** — accuracy, confusion matrix, classification report
9. **Visualise Results** — accuracy comparison, ROC curves, feature importance

---

## Models Used

| Model               | Type              | Notes                              |
|---------------------|-------------------|------------------------------------|
| Logistic Regression | Linear classifier | Simple baseline model, needs scaling |
| Decision Tree       | Tree-based        | Easy to interpret, max_depth=6     |
| Random Forest       | Ensemble          | Best performer, 100 trees          |

---

## Feature Engineering
Two new features were created from existing columns:

- `HouseAge` = 2025 − YearBuilt
- `YearsSinceRemodel` = 2025 − YearRemodAdd

These are more meaningful to the model than raw year values.

---

## Results

| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | ~79%     |
| Decision Tree       | ~79%     |
| Random Forest       | ~88%     |

Random Forest performed best because it combines multiple decision trees and reduces overfitting by averaging their predictions.

---

## Output Files

| File                     | Description                          |
|--------------------------|--------------------------------------|
| `eda_visualisations.png` | 6 EDA plots                          |
| `model_evaluation.png`   | Accuracy bar, ROC curves, feature importance |
| `confusion_matrices.png` | Confusion matrix for all 3 models    |

---

## How to Run

1. Make sure `train.csv` is in the same folder as `house_price_prediction.py`
2. Install required libraries:
```
pip install pandas numpy matplotlib seaborn scikit-learn
```
3. Run the script:
```
python house_price_prediction.py
```
4. The 3 output PNG files will be generated in the same folder.

---

## Libraries Used

- `pandas` — data loading and manipulation
- `numpy` — numerical operations
- `matplotlib` & `seaborn` — data visualisation
- `scikit-learn` — machine learning models and evaluation

---

## Key Observations

- **Overall Quality** (OverallQual) is the single strongest predictor of house price.
- **Living Area** (GrLivArea) — bigger houses tend to cost more, as expected.
- **House Age** — newer houses generally fall in the High price category.
- **Neighborhood** also plays a role since location affects price significantly.
- Random Forest outperforms simpler models because it handles non-linear relationships better and is less affected by noisy features.
