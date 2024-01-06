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
    # print(filepath)
    # print(filepath.split('\\'))
    user_name = filepath.split('\\')[1]
    print(f"Processing {filepath} dataset...")

    # Read in data file as a pandas dataframe
    data = pd.read_csv(filepath)

    # Extract full dataset input 'latency' and 'scale' as X, and 'throughput' as Y.
    X = data[['latency', 'scale']]
    data["performance"] = 10*data['throughput'] - data['avg_osd'] - data['avg_target_error']
    Y = data["performance"]



    # Total number of data points
    n = len(data)

    # Initialize lists to store model accuracies and corresponding n_train values
    r2_scores = []
    mse_scores = []
    full_mse_scores = []
    n_train_mse = []
    n_train_full_mse = []
    optimal_match_rate = []

    for n_train in range(2, n-2):
        # Split, train and predict as in the original script
        X_train, X_test, Y_train, Y_test = even_train_split(data, n_train, y_metric="performance")
        # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=n_train/n)
        poly = PolynomialFeatures(degree=2)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        model = LinearRegression()
        model.fit(X_train_poly, Y_train)
        Y_test_pred = model.predict(X_test_poly)
        r2_scores.append(r2_score(Y_test, Y_test_pred))
        mse = mean_squared_error(Y_test, Y_test_pred)
        if mse < 5000:
            n_train_mse.append(n_train)
            mse_scores.append(mse)
        Y_pred = model.predict(poly.transform(X))
        data['Y_pred'] = Y_pred
        # Calculate optimal scale match rate
        optimal_scale_ref = data.loc[data.groupby('latency')['performance'].idxmax()][['latency', 'scale']]
        optimal_scale_pred = data.loc[data.groupby('latency')['Y_pred'].idxmax()][['latency', 'scale']]

        # Step 2: Merge the results on 'latency'
        merged_data = pd.merge(optimal_scale_ref, optimal_scale_pred, on='latency', suffixes=('_ref', '_pred'))

        # Step 3: Count the number of matches
        matches = (merged_data['scale_ref'] == merged_data['scale_pred']).sum()

        optimal_match_rate.append(matches / len(optimal_scale_ref))
        
        full_mse = mean_squared_error(Y, Y_pred)
        if full_mse < 5000:
            n_train_full_mse.append(n_train)
            full_mse_scores.append(full_mse)

    # Store results from this dataset
    all_results[user_name] = {
        'n_train_mse': list(n_train_mse),
        'n_train_full_mse': list(n_train_full_mse),
        'full_mse_scores': full_mse_scores,
        'mse_scores': mse_scores,
        'n_train_all': range(2, n-2),
        'match_rate': optimal_match_rate
    }

# # Plotting the results for all datasets
# fig, axes = plt.subplots(2, 1, figsize=(12, 12))
# fig.suptitle("2nd Order PR Modeling performance Metric")
# for user_name, results in all_results.items():
#       # Extract user name from the filepath
#     axes[0].plot(results['n_train_full_mse'], results['full_mse_scores'], marker='o', label=user_name)
#     axes[1].plot(results['n_train_mse'], results['mse_scores'], marker='o', label=user_name)

# axes[0].set_title('MSE on whole dataset for all users')
# axes[0].set_xlabel('Number of Training Points (n_train)')
# axes[0].set_ylabel('Model Accuracy (MSE Score)')
# axes[0].grid(True)
# axes[0].legend()

# axes[1].set_title('MSE on test set for all users')
# axes[1].set_xlabel('Number of Training Points (n_train)')
# axes[1].set_ylabel('Model Accuracy (MSE Score)')
# axes[1].grid(True)
# axes[1].legend()

# plt.savefig('figures/poly2_evensplit_acc_performance_all_users(2).png')
# plt.show()

plt.figure(figsize=(8, 8))
plt.title("Optimal Scaling Factor Prediction Rate")
for user_name, results in all_results.items():
    plt.plot(results['n_train_all'], results['match_rate'], marker='o', label=user_name)
plt.xlabel("Training Points")
plt.ylabel("Percentage of Correct Predictions")
plt.legend()
plt.savefig("figures/optimal_scale_match_rate.png")
plt.show()
