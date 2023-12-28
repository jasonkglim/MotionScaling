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
model_accuracies = []
n_train_values = []

# Repeat the process for n_train = 1 to n - 2
for n_train in range(20, 21):
    # Split the data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=n_train/n)
    print(X_train)
    # Perform polynomial regression
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_poly, Y_train)
    
    # Predict and calculate the model accuracy on the test set
    Y_pred = model.predict(X_train_poly)
    accuracy = r2_score(Y_test, Y_pred)

    data["Y_pred"] = Y_pred
    
    # Store the results
    model_accuracies.append(accuracy)
    n_train_values.append(n_train)

    # Plotting with the annotate function to highlight training points
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))

    # Original data heatmap with all points highlighted (now all are training points)
    original_data = data.pivot(
        index='latency', columns='scale', values='throughput'
    )
    sns.heatmap(original_data, cmap='viridis', ax=ax[0])
    annotate(ax[0], original_data, X_train, color='red')
    ax[0].set_title('Original Data')
    ax[0].set_xlabel('Scale')
    ax[0].set_ylabel('Latency')

    # Full predicted data heatmap (prediction on the entire dataset)
    sns.heatmap(full_predicted_data, cmap='viridis', ax=ax[1])
    annotate(ax[1], full_predicted_data, X_train_full, color='red')
    ax[1].set_title('Predicted Data with All Points as Training')
    ax[1].set_xlabel('Scale')
    ax[1].set_ylabel('Latency')

    plt.show()

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(n_train_values, model_accuracies, marker='o')
plt.title('Model Accuracy as a Function of Training Set Size')
plt.xlabel('Number of Training Points (n_train)')
plt.ylabel('Model Accuracy (R^2 Score)')
plt.grid(True)
plt.show()