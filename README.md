# MatSci176_LiMetal

README: Data Cleaning Script

Overview
This script reads in the CALiSol-23 Dataset CSV file, performs a series of data cleaning and preprocessing steps, and then writes out a cleaned CSV. 


How to Use

Install Dependencies
Make sure you have Python 3.7+ and the following libraries installed:
pip install pandas numpy scikit-learn


Detailed Steps

Load the dataset
Reads the CSV file specified by file_path into a Pandas DataFrame named df.

Remove duplicate rows, check and handle missing values
Ensures there are no exact-duplicate entries in the dataset.
Prints a summary of how many rows are missing data for each column. Drops rows that lack essential values for conductivity (k) or temperature (T), as these are critical for the analysis.

Filter only mol/kg data
Keeps rows whose concentration units (c units) are explicitly “mol/kg,” discarding those in other units. This ensures a uniform measure for concentration.

Convert Data Types
Verifies that the temperature (T) and conductivity (k) columns are numeric.
Converts them if they are stored as strings.

Encode Categorical Columns
If the “salt” column is present, the code uses a LabelEncoder to transform each unique salt name into an integer code.
This step is critical for algorithms that need numeric rather than string inputs.






README: Random Forest Model Training and Evaluation


Overview

This script takes a cleaned electrolyte dataset (e.g., the CALiSol-23 dataset) and performs the following steps:
Defines features (temperature, concentration, salt type, and various solvents).
Splits the data into training and test sets.
Trains a Random Forest Regressor to predict ionic conductivity (k).
Evaluates the model using metrics such as MSE, RMSE, and R².
Displays feature importance to understand which variables matter most.
Generates two plots: a parity plot (Actual vs. Predicted) and a bar chart of feature importances.

How to Use

Install Dependencies
Make sure you have Python 3.7+ and the following libraries installed:
pip install pandas numpy scikit-learn matplotlib

Detailed Steps

Imports
pandas, numpy for data manipulation.
sklearn.model_selection for splitting data into train/test sets.
sklearn.ensemble for the Random Forest Regressor.
sklearn.metrics for regression performance metrics.
matplotlib for plotting.

Data loading
Reads the cleaned CSV (already preprocessed in a separate script).

Feature and target definition
A list of feature_cols specifies which columns to include as inputs (e.g., temperature, salt_label, and any relevant solvents).
The target y is ionic conductivity ("k").

Data splitting
Uses an 80/20 split (test_size=0.2) with a fixed random seed (random_state=42) to ensure reproducible results.

Random forest training
Initializes a RandomForestRegressor with 100 trees (n_estimators=100) and trains it on the training set (X_train, y_train).

Model evaluation
Predicts conductivity on the test set (X_test) and calculates:
MSE (Mean Squared Error)
RMSE (Root Mean Squared Error)
R² (coefficient of determination)

Feature importances
Prints the relative importance of each feature in the final model, sorted highest to lowest.

Plotting
Predicted vs. Actual: A scatter plot of y_test vs. y_pred, including a reference line y = x.
Feature Importance Bar Chart: Shows the same importances in a more visual format.

Interpreting Results
High R² and low RMSE suggest that the Random Forest model captures the majority of variability in conductivity.
Feature importance reveals which conditions or chemical components (e.g., temperature, salt type, solvent fractions) most strongly influence conductivity.
The parity plot allows you to spot whether the model tends to over- or under-predict at certain conductivity ranges.


