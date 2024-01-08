# Modeling performance with Gaussian Process Regression

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import seaborn as sns
from utils import stratified_sample, annotate
import glob

# Pattern to match the data files
file_pattern = "data_files/user_*/metric_df.csv"

# Initialize a dictionary to store results from each dataset
all_datasets = {}

# Loop through each file that matches the file pattern
for filepath in glob.glob(file_pattern):
    print(f"Reading in {filepath}...")

    # Read in data file as a pandas dataframe
    data = pd.read_csv(filepath)

    # Extracting features and target variable
    X = data[['latency', 'scale']]
    Y = data['throughput']

    mse_scores = []
    full_mse_scores = []

    # n_train = 10
    n = len(data)
    n_train_values = range(2, n-2)
    for n_train in n_train_values:
        # Splitting the data into training and testing sets
        # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=n_train/n)
        train_set, test_set = stratified_sample(data, n_train)
        X_train = train_set[['latency', 'scale']]
        X_test = test_set[['latency', 'scale']]
        Y_train = train_set['throughput']
        Y_test = test_set['throughput']
        #print(X_train)
        #print(X_test)

        # Define the Gaussian Process kernel
        kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0, 1.0], (1e-2, 1e2))

        # Initialize the Gaussian Process Regressor with the chosen kernel
        gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)

        # Fit the model to the training data
        gp_model.fit(X_train, Y_train)
        # print(gp_model.kernel_)

        # ### Predict on a denser range of test inputs
        # # Generate a grid of test inputs
        # latency_range = np.arange(0.0, 0.76, 0.01)
        # scale_range = np.arange(0.075, 1.025, 0.025)

        # # Create a meshgrid from the input ranges
        # latency_grid, scale_grid = np.meshgrid(latency_range, scale_range)
        # test_inputs = np.c_[latency_grid.ravel(), scale_grid.ravel()]
        # test_inputs = np.round(test_inputs, 3)

        # Predict and measure accuracy on the test inputs
        Y_test_pred, Y_test_pred_std = gp_model.predict(X_test, return_std=True)
        mse = mean_squared_error(Y_test, Y_test_pred)
        mse_scores.append(mse)
        # print(f"Mean Squared Error on the test set: {mse}")

        # Predict over whole dataset for visualization
        Y_full_pred, Y_full_pred_std = gp_model.predict(X, return_std=True)
        data["throughput_pred"] = Y_full_pred
        data["throughput_pred_std"] = Y_full_pred_std
        full_mse_score = mean_squared_error(Y, Y_full_pred)
        full_mse_scores.append(full_mse_score)

        # Store results from this dataset
        all_results[filepath] = {
            'n_train_values': list(n_train_values),
            'full_mse_scores': full_mse_scores,
            'mse_scores': mse_scores
        }

        # # Reshape the predictions and uncertainties to match the grid shape
        # y_pred = y_pred.reshape(latency_grid.shape)
        # sigma = sigma.reshape(latency_grid.shape)

        # # Create a DataFrame for the predicted values
        # pred_df = pd.DataFrame({
        #     'latency': test_inputs[:, 0].flatten(),
        #     'scale': test_inputs[:, 1].flatten(),
        #     'mean': y_pred.flatten(),
        #     'std': sigma.flatten()
        # })

        # # Evaluate the model performance (you can use other metrics as well)
        # mse = mean_squared_error(y_test, y_pred)
        # print(f"Mean Squared Error on the test set: {mse}")



        # # Visualization
        # fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        # # Plot original data
        # # Plot heatmap for average movement time
        # heatmap_TP = data.pivot(
        #         index='latency', columns='scale', values='throughput')
        # #print(heatmap_TP)
        # sns.heatmap(heatmap_TP, ax=axes[0, 0], cmap="YlGnBu", annot=True, fmt='.3g')
        # axes[0, 0].set_title('Throughput vs. Latency and Scale')
        # annotate(axes[0, 0], heatmap_TP, X_train)

        # heatmap_pred_TP = data.pivot(
        #     index='latency', columns='scale', values='throughput_pred')
        # sns.heatmap(heatmap_pred_TP, ax=axes[0, 1], cmap="YlGnBu", annot=True, fmt='.3g')
        # axes[0, 1].set_title('Mean Predicted Throughput vs. Latency and Scale')
        # annotate(axes[0, 1], heatmap_pred_TP, X_train)

        # heatmap_pred_std = data.pivot(
        #     index='latency', columns='scale', values='throughput_pred_std')
        # sns.heatmap(heatmap_pred_std, ax=axes[1, 0], cmap="YlGnBu", annot=True, fmt='.3g')
        # axes[1, 0].set_title('Throughput Prediction STD vs. Latency and Scale')
        # annotate(axes[1, 0], heatmap_pred_std, X_train)

        # data["residual"] = np.abs(data["throughput"] - data["throughput_pred"])
        # heatmap_res = data.pivot(
        #     index='latency', columns='scale', values='residual')
        # sns.heatmap(heatmap_res, ax=axes[1, 1], cmap="YlGnBu", annot=True, fmt='.3g')
        # axes[1, 1].set_title('Throughput Prediction Residuals vs. Latency and Scale')
        # annotate(axes[1, 1], heatmap_res, X_train)

        # plt.tight_layout()
        # # plt.savefig('figures/gpr_results.png')
        # plt.show()

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

plt.savefig('figures/gpr_model_acc_vs_n_train_evensplit_all_users.png')
plt.show()
