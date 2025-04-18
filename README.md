# 🏠 Housing Price Prediction

This project aims to predict housing prices using machine learning models and analyze the importance of housing features. The dataset contains structured information about real estate properties including area, number of rooms, location factors, and furnishing status.

---

## 📁 Project Structure

```text
housing-price-prediction/
│
├── data/
│   ├── Housing.csv                  # Original dataset
│   └── housing_cleaned.csv          # Cleaned dataset with additional features
│
├── notebooks/
│   ├── 01_data_preparation.ipynb    # Data cleaning, encoding, outlier detection
│   └── 02_model_training.ipynb      # Model training, evaluation, feature importance
│
├── requirements.txt                 # Required Python libraries
└── README.md                        

```

## 📊 Dataset Overview

The dataset includes:

- `area`, `bedrooms`, `bathrooms`, `stories`
- `mainroad`, `guestroom`, `basement`, etc.
- `furnishingstatus`: encoded in multiple formats (ordinal and one-hot)
- Engineered features:
  - `area_per_bedroom`
  - `area_per_bathroom`
  - `is_fully_equipped`

---

##  Workflow Summary

### 1. Data Cleaning & Encoding
- Converted object columns to numeric/boolean
- Used one-hot and ordinal encoding strategies
- Detected and removed outliers using Isolation Forest
- Created engineered features


### 2. Model Training
Trained and evaluated the following models:

| Model                          | Encoding Used  |
|--------------------------------|----------------|
| Linear Regression              | One-Hot        |
| Ridge Regression               | One-Hot        |
| Random Forest Regressor        | Ordinal        |
| Gradient Boosting Regressor    | Ordinal        |
| XGBoost Regressor              | Ordinal        |

All models were compared using:
- **RMSE**
- **MAE**
- **R² Score**
- **Actual vs Predicted** scatter plots

### 3. Feature Importance
- Coefficients (linear models)
- Feature importances (tree models)


---

## 📈 Key Insights

- `area` was the most influential feature across all models
- Engineered features (like `area_per_bathroom`) boosted performance in tree models
- Furnishing and amenities like `airconditioning` played a significant role in linear models


---

## Next Steps

- Fine-tune hyperparameters
- Use SHAP values for explainability
- Explore ensemble and stacking models
- Deploy a trained model as an API

---
