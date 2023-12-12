# Modeling performance with Gaussian Process Regression

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Assuming 'metric_df' is your DataFrame with columns 'latency', 'scale', and 'performance'
# You might need to preprocess the data based on your specific requirements.

# Read in data
metric_df = pd.read_csv('data_files/user_jason')

# Extracting features and target variable
X = metric_df[['latency', 'scale']]
y = metric_df['performance']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train)
print(X_test)

# # Define the Gaussian Process kernel
# kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0, 1.0], (1e-2, 1e2))

# # Initialize the Gaussian Process Regressor with the chosen kernel
# gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)

# # Fit the model to the training data
# gp_model.fit(X_train, y_train)

# # Predict on the test set
# y_pred, sigma = gp_model.predict(X_test, return_std=True)

# # Evaluate the model performance (you can use other metrics as well)
# mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error on the test set: {mse}")

# # Visualize the results if needed
# # (Note: Visualization can be complex if the data has more than 2 dimensions)
