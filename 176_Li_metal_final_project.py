#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 1. Load the dataset
#file_path = r"C:\Users\fernn\Dropbox\untitled folder\Stanford courses\CHEMENG_177\CALiSol-23 Dataset.csv"
df = pd.read_csv("CALiSol-23 Dataset.csv")


# 2. Remove duplicate rows, if any
df.drop_duplicates(inplace=True)


# 3. Check for missing values
missing_summary = df.isna().sum()
print("Missing values per column:\n", missing_summary)


# drop rows missing 'k' (conductivity) or 'T'
df.dropna(subset=["k", "T"], inplace=True)


# 4. Filter data to only include rows already in mol/kg
df = df[df["c units"] == "mol/kg"]


# 5. Convert data types
if df["T"].dtype == object:
    df["T"] = pd.to_numeric(df["T"], errors="coerce")
if df["k"].dtype == object:
    df["k"] = pd.to_numeric(df["k"], errors="coerce")


# 6. Encode categorical columns (salt column)
if "salt" in df.columns:
    le = LabelEncoder()
    df["salt_label"] = le.fit_transform(df["salt"])


# 7. Final check: shape and a preview
print("Shape of cleaned dataset:", df.shape)
print(df.head())


# 8. Save the cleaned DataFrame
#output_path = r"CALiSol-23 Dataset(cleaned).csv"
df.to_csv("CALiSol-23 Dataset(cleaned).csv", index=False)
print(f"Cleaned dataset saved to {'CALiSol-23 Dataset(cleaned)'}")

# For train/test split and model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load cleaned dataset
#file_path = r"C:\Users\fernn\Dropbox\untitled folder\Stanford courses\CHEMENG_177\CALiSol-23 Dataset(cleaned).csv"
df = pd.read_csv('CALiSol-23 Dataset(cleaned).csv')


# 2. Define features (X) and target (y)
feature_cols = [
    "T",   # Temperature
    "c",   # Concentration (mol/kg)
    "salt_label",   # Encoded salt
    # All relevant solvents
    "EC",
    "PC",
    "DMC",
    "Ethylbenzene",
    "Ethylmonoglyme",
    "Benzene",
    "g-Butyrolactone",
    "Cumene",
    "Propylsulfone",
    "Pseudocumeme",
    "TEOS",
    "m-Xylene",
    "o-Xylene"
]


# Ensure these columns actually exist in df; if not, remove from feature_cols
existing_features = [col for col in feature_cols if col in df.columns]
missing_features = set(feature_cols) - set(existing_features)
if missing_features:
    print(f"Warning: The following feature columns do not exist in the dataset and will be excluded: {missing_features}")

X = df[existing_features]
y = df["k"]   # Target: ionic conductivity


# 3. Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 4. Initialize and train the Random Forest Regressor
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    random_state=42
)
rf.fit(X_train, y_train)


# 5. Make predictions on the test set
y_pred = rf.predict(X_test)


# 6. Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nRandom Forest Regression Results:")
print(f"  MSE:  {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R²:   {r2:.4f}")


# 7. Inspect feature importance
importances = rf.feature_importances_
for col, imp in sorted(zip(existing_features, importances), key=lambda x: x[1], reverse=True):
    print(f"{col}: {imp:.3f}")

    
# Generate plots
import matplotlib.pyplot as plt

# y_test: actual conductivity values
# y_pred: predicted conductivity values from my model
plt.scatter(y_test, y_pred, alpha=0.7)  # alpha=0.7 makes points slightly transparent
plt.xlabel("Actual Conductivity")
plt.ylabel("Predicted Conductivity")
plt.title("Predicted vs. Actual Conductivity")


# Plot a reference line y=x for comparison
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], '--')  # dashed line

plt.show()
import matplotlib.pyplot as plt
import numpy as np


# 1. Retrieve the importances and sort them
idx_sorted = np.argsort(importances)  # ascending order

# 2. Create a bar chart
plt.bar(range(len(idx_sorted)), importances[idx_sorted])

# 3. Label the x-axis with feature names, rotated for readability
plt.xticks(
    range(len(idx_sorted)),
    [existing_features[i] for i in idx_sorted],
    rotation=45,
    ha="right"
)

# 4. Label axes and give a title
plt.ylabel("Feature Importance")
plt.title("Random Forest Feature Importances")

# 5. Tight layout to avoid label cutoff, then show
plt.tight_layout()
plt.show()


# In[ ]:





