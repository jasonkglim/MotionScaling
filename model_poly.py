### script for trying models
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Function to add red border to maximum value in each row
def annotate(ax, data, points, color='red'):
    """
    Annotate the heatmap with a rectangle around the training points.

    Parameters:
    ax (matplotlib.axes._subplots.AxesSubplot): The axes on which to annotate.
    data (pandas.DataFrame): The dataframe used for the heatmap, indexed by 'Latency' and 'Scale'.
    points (pandas.DataFrame): The training points to highlight.
    color (str): The color of the rectangles.
    """
    for index, row in points.iterrows():
        # Find the position of the training point in the heatmap
        row_idx = data.index.get_loc(row['latency'])  # Adjusted for correct key
        col_idx = data.columns.get_loc(row['scale'])  # Adjusted for correct key
        
        # Add a rectangle around the corresponding cell in the heatmap
        ax.add_patch(plt.Rectangle((col_idx, row_idx), 1, 1, fill=False, edgecolor=color, lw=3))

# Read in data file as a pandas dataframe
data = pd.read_csv('data_files/user_jason/metric_df.csv')

# Extract 'latency' and 'scale' as X, and 'throughput' as Y.
X = data[['latency', 'scale']]
Y = data['throughput']

# Total number of data points
n = len(data)

# Initialize lists to store model accuracies and corresponding n_train values
r2_scores = []
mse_scores = []
n_train_values = []

# Repeat the process for n_train = 1 to n - 2
for n_train in range(1, 26):
    # Split the data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=n_train/n)
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

    # Store the results
    n_train_values.append(n_train)

    if mse_scores[-1] > 100:

        # Plotting with the annotate function to highlight training points
        fig, ax = plt.subplots(1, 3, figsize=(18, 8))

        # Original data heatmap with all points highlighted (now all are training points)
        original_data = data.pivot(
            index='latency', columns='scale', values='throughput'
        )
        sns.heatmap(original_data, cmap='YlGnBu', ax=ax[0], annot=True)
        annotate(ax[0], original_data, X_train, color='red')
        ax[0].set_title('Original Data')
        ax[0].set_xlabel('Scale')
        ax[0].set_ylabel('Latency')

        # Full predicted data heatmap (prediction on the entire dataset)
        predicted_data = data.pivot(
            index='latency', columns='scale', values='Y_pred'
        )
        sns.heatmap(predicted_data, cmap='YlGnBu', ax=ax[1], annot=True)
        annotate(ax[1], predicted_data, X_train, color='red')
        ax[1].set_title('Predicted Data')
        ax[1].set_xlabel('Scale')
        ax[1].set_ylabel('Latency')

        # Plot residuals
        data["residual"] = np.abs(data["throughput"] - data["Y_pred"])
        residual = data.pivot(
            index='latency', columns='scale', values='residual'
        )
        sns.heatmap(residual, cmap='YlGnBu', ax=ax[2], annot=True)
        annotate(ax[2], residual, X_train, color='red')
        ax[2].set_title('Residuals')
        ax[2].set_xlabel('Scale')
        ax[2].set_ylabel('Latency')

        plt.tight_layout()
        plt.show()

print(mse_scores)

# Plotting the results
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].plot(n_train_values, r2_scores, marker='o')
axes[0].set_title('R^2')
axes[0].set_xlabel('Number of Training Points (n_train)')
axes[0].set_ylabel('Model Accuracy (R^2 Score)')
axes[0].grid(True)

axes[1].plot(n_train_values, mse_scores, marker='o')
axes[1].set_title('MSE')
axes[1].set_xlabel('Number of Training Points (n_train)')
axes[1].set_ylabel('Model Accuracy (MSE Score)')
axes[1].grid(True)

plt.savefig('figures/poly2_model_acc_vs_n_train(1).png')
plt.show()