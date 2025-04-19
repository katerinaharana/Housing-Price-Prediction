#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[25]:


# Step 2: Confirm you are in the right place (optional)
print("Now in:", os.getcwd())


# In[29]:


# Go from /notebooks to /data
os.chdir('./data')

# Confirm you're in the right place (you should see the CSV here)
print("Now in:", os.getcwd())





# In[ ]:


engineered_features = ['area_per_bedroom', 'area_per_bathroom', 'is_fully_equipped']
# One-hot: for linear models
X_onehot = df.drop(columns=['price', 'furnishingstatus_ordinal'])

# Ordinal: for tree models
furnish_cols = [col for col in df.columns if col.startswith('furnishingstatus_') and col != 'furnishingstatus_ordinal']
X_ordinal = df.drop(columns=['price'] + furnish_cols)



# In[ ]:


y = df['price']


# In[ ]:


X_ord_train, X_ord_test, y_train, y_test = train_test_split(X_ordinal, y, test_size=0.2, random_state=42)
X_ohe_train, X_ohe_test = train_test_split(X_onehot, test_size=0.2, random_state=42)


# In[ ]:


scaler = StandardScaler()
X_ohe_train_scaled = scaler.fit_transform(X_ohe_train)
X_ohe_test_scaled = scaler.transform(X_ohe_test)


# In[ ]:


models = {
    "Linear Regression (One-Hot)": (LinearRegression(), X_ohe_train_scaled, X_ohe_test_scaled),
    "Ridge Regression (One-Hot)": (Ridge(alpha=1.0), X_ohe_train_scaled, X_ohe_test_scaled),
    "Random Forest (Ordinal)": (RandomForestRegressor(random_state=42), X_ord_train, X_ord_test),
    "Gradient Boosting (Ordinal)": (GradientBoostingRegressor(random_state=42), X_ord_train, X_ord_test),
    "XGBoost (Ordinal)": (XGBRegressor(random_state=42, verbosity=0, n_estimators=100), X_ord_train, X_ord_test)}


# In[ ]:


results = []

for name, (model, X_train, X_test) in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    results.append({
        "Model": name,
        "RMSE": round(rmse, 2),
        "R² Score": round(r2, 4),
        "MAE": round(mae,2)
    })


# In[ ]:


results_df = pd.DataFrame(results).sort_values(by="RMSE")
print(results_df)


# In[ ]:


for name, (model, X_train, X_test) in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # Scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds, alpha=0.6)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color='red', linestyle='--', label='Perfect Prediction'
    )
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"{name}\nRMSE: {rmse:,.0f}, MAE: {mae:,.0f}, R²: {r2:.3f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# In[ ]:


# 📊 Feature importance for tree-based models
tree_models = {
    "Random Forest": (RandomForestRegressor(random_state=42), X_ordinal),
    "Gradient Boosting": (GradientBoostingRegressor(random_state=42), X_ordinal),
    "XGBoost": (XGBRegressor(n_estimators=100, max_depth=3, random_state=42, verbosity=0), X_ordinal)
}

for name, (model, X) in tree_models.items():
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    top = importances.sort_values(ascending=False).head(10)

    print(f"\n🔹 {name} - Top 10 Features 🔹")
    print(importances.head(10))

    top.plot(kind='barh', title=f"Top Features - {name}", color='darkgreen')
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# In[ ]:


from sklearn.linear_model import LinearRegression, Ridge
import pandas as pd

# Define models and labels
linear_models = {
    "Ridge Regression": Ridge(alpha=1.0),
    "Linear Regression": LinearRegression()
}

# Input: X_ohe_train (scaled), y_train
for name, model in linear_models.items():
    model.fit(X_ohe_train, y_train)

    # Get coefficients
    coefs = pd.Series(model.coef_, index=X_ohe_train.columns)
    coefs_sorted = coefs.sort_values(key=abs, ascending=False)

    # Display top 10 most influential features
    print(f"\n🔍 Top 10 most influential features in {name}:")
    print(coefs_sorted.head(10))

    # Optional bar plot
    coefs_sorted.head(10).plot(kind='barh', title=f"Top Features - {name}")
    plt.gca().invert_yaxis()
    plt.xlabel("Coefficient Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# In[ ]:


get_ipython().system('jupyter nbconvert --to script 02_model_training.ipynb --output-dir=../notebooks')

