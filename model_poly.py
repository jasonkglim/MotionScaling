### script for trying models
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils import stratified_sample, annotate, even_train_split, annotate_extrema


# Read in data file as a pandas dataframe
data = pd.read_csv('data_files/user_jason/metric_df.csv')

# Extract full dataset input 'latency' and 'scale' as X, and 'throughput' as Y.
X = data[['latency', 'scale']]
data["performance"] = 10*data["throughput"] - data["avg_osd"] - data['avg_target_error']
Y = data['performance']

# Total number of data points
n = len(data)

# Initialize lists to store model accuracies and corresponding n_train values
r2_scores = []
mse_scores = []
full_mse_scores = []
optimal_match_rate = []
optimal_scale_errors = []
# n_train_values = []

# Repeat the process for n_train = 1 to n - 2
n_train_values = [25] #range(2, n-2) # [4, 8, 12, 16, 20, 24]
for n_train in n_train_values:
	# Split the data into training and test sets
	# train_set, test_set = stratified_sample(data, n_train)
	# X_train = train_set[['latency', 'scale']]
	# X_test = test_set[['latency', 'scale']]
	# Y_train = train_set['throughput']
	# Y_test = test_set['throughput']
	# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=n_train/n)
	X_train, X_test, Y_train, Y_test = even_train_split(data, n_train, y_metric="performance")
	# print("train\n", X_train)
	# print("\ntest\n", X_test)
	# Perform polynomial regression
	poly = PolynomialFeatures(degree=2)
	X_train_poly = poly.fit_transform(X_train)
	X_poly = poly.transform(X)
	X_test_poly = poly.transform(X_test)

	# Train model on training set
	model = LinearRegression()
	model.fit(X_train_poly, Y_train)

	# evaluate model accuracy on the test set
	Y_test_pred = model.predict(X_test_poly)
	r2_scores.append(r2_score(Y_test, Y_test_pred))
	mse_scores.append(mean_squared_error(Y_test, Y_test_pred))

	# Predict over whole dataset for visualization
	Y_pred = model.predict(X_poly)
	data["Y_pred"] = Y_pred
	full_mse_scores.append(mean_squared_error(Y, Y_pred))
	#if mse_scores[-1] > 10:

	# Predict over dense inputs
	latency_range = np.array(data['latency'].unique())# np.arange(0.0, 0.76, 0.01)
	scale_range = np.arange(0.075, 1.025, 0.025)

	# Create a meshgrid from the input ranges
	latency_grid, scale_grid = np.meshgrid(latency_range, scale_range)
	X_dense = np.c_[latency_grid.ravel(), scale_grid.ravel()]
	X_dense = np.round(X_dense, 3)
	X_dense_poly = poly.transform(X_dense)

	Y_pred_dense = model.predict(X_dense_poly)

	dense_df = pd.DataFrame({
            'latency': X_dense[:, 0].flatten(),
            'scale': X_dense[:, 1].flatten(),
            'Y_pred_dense': Y_pred_dense.flatten()
        })
	
	optimal_scale_dense = dense_df.loc[dense_df.groupby('latency')['Y_pred_dense'].idxmax()][['latency', 'scale']]
	optimal_scale_ref = data.loc[data.groupby('latency')['performance'].idxmax()][['latency', 'scale']]
	optimal_scale_pred = data.loc[data.groupby('latency')['Y_pred'].idxmax()][['latency', 'scale']]
	print(optimal_scale_ref)
	print(optimal_scale_pred)
	print(optimal_scale_dense)

	# Merge the results on 'latency'
	merged_ref_pred = pd.merge(optimal_scale_ref, optimal_scale_pred, 
						on='latency', suffixes=('_ref', '_pred'))
	
	merged_ref_dense = pd.merge(optimal_scale_ref, optimal_scale_dense, 
						on='latency', suffixes=('_ref', '_dense'))
	

	# Count the number of matches
	matches = (merged_ref_pred['scale_ref'] == merged_ref_pred['scale_pred']).sum()
	scale_error = np.abs(merged_ref_dense['scale_ref'] - merged_ref_dense['scale_dense']).mean()

	optimal_match_rate.append(matches / 4)
	optimal_scale_errors.append(scale_error)

	# Plotting with the annotate function to highlight training points
	cond = False # n_train in [25]
	if cond:
		fig, ax = plt.subplots(1, 3, figsize=(18, 8))

		# Original data heatmap with all points highlighted (now all are training points)
		original_data = data.pivot(
			index='latency', columns='scale', values='performance'
		)
		sns.heatmap(original_data, cmap='YlGnBu', ax=ax[0], annot=True)
		# annotate(ax[0], original_data, X_train, color='green')
		ax[0].set_title('Original Data Performance Metric')
		ax[0].set_xlabel('Scale')
		ax[0].set_ylabel('Latency')
		annotate_extrema(original_data.values, ax[0])

		# Full predicted data heatmap (prediction on the entire dataset)
		predicted_data = data.pivot(
			index='latency', columns='scale', values='Y_pred'
		)
		sns.heatmap(predicted_data, cmap='YlGnBu', ax=ax[1], annot=True)
		annotate(ax[1], predicted_data, X_train, color='green')
		ax[1].set_title('Predicted Data')
		ax[1].set_xlabel('Scale')
		ax[1].set_ylabel('Latency')
		annotate_extrema(predicted_data.values, ax[1])

		dense_pred_data = dense_df.pivot(
			index='latency', columns='scale', values='Y_pred_dense'
		)
		sns.heatmap(dense_pred_data, cmap='YlGnBu', ax=ax[2], annot=True)
		# annotate(ax[1], dense_pred_data, X_train, color='green')
		ax[2].set_title('Predicted Data')
		ax[2].set_xlabel('Scale')
		ax[2].set_ylabel('Latency')
		annotate_extrema(dense_pred_data.values, ax[2])

		# # Plot residuals
		# data["residual"] = np.abs(data["performance"] - data["Y_pred"])
		# residual = data.pivot(
		# 	index='latency', columns='scale', values='residual'
		# )
		# sns.heatmap(residual, cmap='YlGnBu', ax=ax[2], annot=True)
		# annotate(ax[2], residual, X_train, color='green')
		# ax[2].set_title('Residuals')
		# ax[2].set_xlabel('Scale')
		# ax[2].set_ylabel('Latency')
		# annotate_extrema(residual.values, ax[2], 'min')

		plt.tight_layout()
		plt.show()

# Plotting the results
fig, axes = plt.subplots(2, 2, figsize=(16, 6))
axes[0, 0].plot(n_train_values, full_mse_scores, marker='o')
axes[0, 0].set_title('MSE on whole dataset')
axes[0, 0].set_xlabel('Number of Training Points (n_train)')
axes[0, 0].set_ylabel('Model Accuracy (R^2 Score)')
#axes[0].grid(True)

axes[0, 1].plot(n_train_values, mse_scores, marker='o')
axes[0, 1].set_title('MSE on test set') 
axes[0, 1].set_xlabel('Number of Training Points (n_train)')
axes[0, 1].set_ylabel('Model Accuracy (MSE Score)')
# axes[1].grid(True)

axes[1, 0].plot(n_train_values, optimal_match_rate, marker='o')
axes[1, 0].set_title('Correct Optimal Scale Predictions')
axes[1, 0].set_xlabel('Number of Training Points (n_train)')
axes[1, 0].set_ylabel('Number of matches')
# axes[2].grid(True)

axes[1, 1].plot(n_train_values, optimal_scale_errors, marker='o')
axes[1, 1].set_title('Avg Error in Optimal Scale Prediction')
axes[1, 1].set_xlabel('Number of Training Points (n_train)')
axes[1, 1].set_ylabel('Avg Error')

plt.tight_layout()
#plt.savefig('figures/poly2_model_optimal_scale_analysis_jason.png')
plt.show()