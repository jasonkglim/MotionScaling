import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from utils import stratified_sample, annotate, even_train_split
import glob  # Importing the glob module to find all the files matching a pattern

# Pattern to match the data files
file_pattern = "data_files/user_*/metric_df.csv"

# Initialize a dictionary to store results from each dataset
all_results = {}

# Loop through each file that matches the file pattern
for filepath in glob.glob(file_pattern):
    print(f"Processing {filepath}...")

    # Read in data file as a pandas dataframe
    data = pd.read_csv(filepath)

    # Extract full dataset input 'latency' and 'scale' as X, and 'throughput' as Y.
    X = data[['latency', 'scale']]
    Y = data['throughput']

    # Total number of data points
    n = len(data)

    # Initialize lists to store model accuracies and corresponding n_train values
    r2_scores = []
    mse_scores = []
    full_mse_scores = []
    n_train_values = range(2, n-2)

    for n_train in n_train_values:
        # Split, train and predict as in the original script
        X_train, X_test, Y_train, Y_test = even_train_split(data, n_train)
        poly = PolynomialFeatures(degree=2)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        model = LinearRegression()
        model.fit(X_train_poly, Y_train)
        Y_test_pred = model.predict(X_test_poly)
        r2_scores.append(r2_score(Y_test, Y_test_pred))
        mse_scores.append(mean_squared_error(Y_test, Y_test_pred))
        Y_pred = model.predict(poly.transform(X))
        full_mse_scores.append(mean_squared_error(Y, Y_pred))

    # Store results from this dataset
    all_results[filepath] = {
        'n_train_values': list(n_train_values),
        'full_mse_scores': full_mse_scores,
        'mse_scores': mse_scores
    }

# Plotting the results for all datasets
fig, axes = plt.subplots(2, 1, figsize=(12, 12))
for filepath, results in all_results.items():
    user_name = filepath.split('/')[1]  # Extract user name from the filepath
    axes[0].plot(results['n_train_values'], results['full_mse_scores'], marker='o', label=user_name)
    axes[1].plot(results['n_train_values'], results['mse_scores'], marker='o', label=user_name)

axes[0].set_title('MSE on whole dataset for all users')
axes[0].set_xlabel('Number of Training Points (n_train)')
axes[0].set_ylabel('Model Accuracy (R^2 Score)')
axes[0].grid(True)
axes[0].legend()

axes[1].set_title('MSE on test set for all users')
axes[1].set_xlabel('Number of Training Points (n_train)')
axes[1].set_ylabel('Model Accuracy (MSE Score)')
axes[1].grid(True)
axes[1].legend()

plt.savefig('figures/poly2_model_acc_vs_n_train_evensplit_all_users.png')
plt.show()
