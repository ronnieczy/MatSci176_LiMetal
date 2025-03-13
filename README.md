<!DOCTYPE html>
<html lang="en">
<body>
    <div class="section">
        <h1>CALiSol-23 Dataset Processing</h1>
        <h2>Overview</h2>
        <p>This script reads in the CALiSol-23 Dataset CSV file, performs a series of data cleaning and preprocessing steps, and then writes out a cleaned CSV.</p>
        <h2>How to Use</h2>
        <h3>Install Dependencies</h3>
        <p>Make sure you have Python 3.7+ and the following libraries installed:</p>
        <pre><code>pip install pandas numpy scikit-learn</code></pre>
        <h3>Detailed Steps</h3>
        <ul>
            <li><strong>Load the dataset</strong>
                <ul>
                    <li>Reads the CSV file specified by <code>file_path</code> into a Pandas DataFrame named <code>df</code>.</li>
                </ul>
            </li>
            <li><strong>Remove duplicate rows, check and handle missing values</strong>
                <ul>
                    <li>Ensures there are no exact-duplicate entries in the dataset.</li>
                    <li>Prints a summary of how many rows are missing data for each column.</li>
                    <li>Drops rows that lack essential values for conductivity (<code>k</code>) or temperature (<code>T</code>), as these are critical for the analysis.</li>
                </ul>
            </li>
            <li><strong>Filter only mol/kg data</strong>
                <ul>
                    <li>Keeps rows whose concentration units (<code>c units</code>) are explicitly "mol/kg," discarding those in other units.</li>
                    <li>This ensures a uniform measure for concentration.</li>
                </ul>
            </li>
            <li><strong>Convert Data Types</strong>
                <ul>
                    <li>Verifies that the temperature (<code>T</code>) and conductivity (<code>k</code>) columns are numeric.</li>
                    <li>Converts them if they are stored as strings.</li>
                </ul>
            </li>
            <li><strong>Encode Categorical Columns</strong>
                <ul>
                    <li>If the "salt" column is present, the code uses a <code>LabelEncoder</code> to transform each unique salt name into an integer code.</li>
                    <li>This step is critical for algorithms that need numeric rather than string inputs.</li>
                </ul>
            </li>
        </ul>
    </div>
    <div class="section">
        <h1>Random Forest Model Training and Evaluation</h1>
        <h2>Overview</h2>
        <p>This script takes a cleaned electrolyte dataset (e.g., the CALiSol-23 dataset) and performs the following steps:</p>
        <ul>
            <li>Defines features (temperature, concentration, salt type, and various solvents).</li>
            <li>Splits the data into training and test sets.</li>
            <li>Trains a Random Forest Regressor to predict ionic conductivity (<code>k</code>).</li>
            <li>Evaluates the model using metrics such as MSE, RMSE, and R².</li>
            <li>Displays feature importance to understand which variables matter most.</li>
            <li>Generates two plots: a parity plot (Actual vs. Predicted) and a bar chart of feature importances.</li>
        </ul>
        <h2>How to Use</h2>
        <h3>Install Dependencies</h3>
        <p>Make sure you have Python 3.7+ and the following libraries installed:</p>
        <pre><code>pip install pandas numpy scikit-learn matplotlib</code></pre>
        <h3>Detailed Steps</h3>
        <ul>
            <li><strong>Imports</strong>
                <ul>
                    <li><code>pandas</code>, <code>numpy</code> for data manipulation.</li>
                    <li><code>sklearn.model_selection</code> for splitting data into train/test sets.</li>
                    <li><code>sklearn.ensemble</code> for the Random Forest Regressor.</li>
                    <li><code>sklearn.metrics</code> for regression performance metrics.</li>
                    <li><code>matplotlib</code> for plotting.</li>
                </ul>
            </li>
            <li><strong>Data loading</strong>
                <ul>
                    <li>Reads the cleaned CSV (already preprocessed in a separate script).</li>
                </ul>
            </li>
            <li><strong>Feature and target definition</strong>
                <ul>
                    <li>A list of <code>feature_cols</code> specifies which columns to include as inputs (e.g., temperature, salt_label, and any relevant solvents).</li>
                    <li>The target <code>y</code> is ionic conductivity ("<code>k</code>").</li>
                </ul>
            </li>
            <li><strong>Data splitting</strong>
                <ul>
                    <li>Uses an 80/20 split (<code>test_size=0.2</code>) with a fixed random seed (<code>random_state=42</code>) to ensure reproducible results.</li>
                </ul>
            </li>
            <li><strong>Random forest training</strong>
                <ul>
                    <li>Initializes a <code>RandomForestRegressor</code> with 100 trees (<code>n_estimators=100</code>) and trains it on the training set (<code>X_train</code>, <code>y_train</code>).</li>
                </ul>
            </li>
            <li><strong>Model evaluation</strong>
                <ul>
                    <li>Predicts conductivity on the test set (<code>X_test</code>) and calculates:
                        <ul>
                            <li>MSE (Mean Squared Error)</li>
                            <li>RMSE (Root Mean Squared Error)</li>
                            <li>R² (coefficient of determination)</li>
                        </ul>
                    </li>
                </ul>
            </li>
            <li><strong>Feature importances</strong>
                <ul>
                    <li>Prints the relative importance of each feature in the final model, sorted highest to lowest.</li>
                </ul>
            </li>
            <li><strong>Plotting</strong>
                <ul>
                    <li><strong>Predicted vs. Actual:</strong> A scatter plot of <code>y_test</code> vs. <code>y_pred</code>, including a reference line y = x.</li>
                    <li><strong>Feature Importance Bar Chart:</strong> Shows the same importances in a more visual format.</li>
                </ul>
            </li>
        </ul>
        <div class="highlight">
            <h3>Interpreting Results</h3>
            <ul>
                <li>High R² and low RMSE suggest that the Random Forest model captures the majority of variability in conductivity.</li>
                <li>Feature importance reveals which conditions or chemical components (e.g., temperature, salt type, solvent fractions) most strongly influence conductivity.</li>
                <li>The parity plot allows you to spot whether the model tends to over- or under-predict at certain conductivity ranges.</li>
            </ul>
        </div>
    </div>
</body>
</html>
