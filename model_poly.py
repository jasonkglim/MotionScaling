### script for trying models
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

def annotate_extrema(data, ax, extrema_type='max'):
    if extrema_type == 'max':
        extrema_index = np.argmax(data, axis=1)
        color = 'red'
    if extrema_type == 'min':
        extrema_index = np.argmin(data, axis=1)
        color = 'orange'
    for i, max_col in enumerate(extrema_index):
        ax.add_patch(plt.Rectangle((max_col, i), 1, 1, fill=False, edgecolor=color, lw=3))

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
for n_train in range(1, n):
    # Split the data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=n_train/n)
    
    # Perform polynomial regression
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_poly, Y_train)
    
    # Predict and calculate the model accuracy on the test set
    Y_pred = model.predict(X_test_poly)
    accuracy = r2_score(Y_test, Y_pred)
    
    # Store the results
    model_accuracies.append(accuracy)
    n_train_values.append(n_train)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(n_train_values, model_accuracies, marker='o')
plt.title('Model Accuracy as a Function of Training Set Size')
plt.xlabel('Number of Training Points (n_train)')
plt.ylabel('Model Accuracy (R^2 Score)')
plt.grid(True)
plt.show()



# for set_num in range(1):
#     # Read in data
# #    set_num = 6
#     user = "jason"
#     data_folder = f"data_files/user_{user}"
#     df = pd.read_csv(f'{data_folder}/metric_df.csv')

#     latency_vals = [0.0, 0.2, 0.4, 0.6, 0.8]
#     scale_vals = [0.4, 0.6, 0.8, 1.0, 1.2]

#     # Extracting the independent variables and the dependent variable
#     X = df[['latency', 'scale']]
#     df['combo'] = 10*df['throughput'] - df['avg_osd'] - df['avg_target_error']
#     y = df['throughput']

#     # Creating polynomial features
#     degrees = [2, 3, 4]  # List of degrees for different models

#     fig, axes = plt.subplots(1, len(degrees) + 1, figsize=(5 * (len(degrees) + 1), 6))  # Adjusting subplots

#     for i, degree in enumerate(degrees):
#         poly = PolynomialFeatures(degree=degree, include_bias=False)
#         X_poly = poly.fit_transform(X)

#         # Fitting the polynomial features to Linear Regression
#         model = LinearRegression()
#         model.fit(X_poly, y)

#         # Calculating Mean Squared Error
#         y_pred = model.predict(X_poly)
#         mse = mean_squared_error(y, y_pred)

#         # Plot surface predicted by model
#         mesh_df = pd.DataFrame([[i, j] for i in latency_vals for j in scale_vals], columns=['latency', 'scale'])
#         X_mesh = mesh_df[['latency', 'scale']]
#         X_mesh_poly = poly.fit_transform(X_mesh)

#         # Predicting the values for the meshgrid
#         predicted_values = model.predict(X_mesh_poly)

#         mesh_df[f'predicted_{degree}'] = predicted_values

#         # Plotting the heatmaps
#         heatmap_predicted = mesh_df.pivot(
#             index='latency', columns='scale', values=f'predicted_{degree}')
#         ax = sns.heatmap(heatmap_predicted, ax=axes[i + 1], annot=True, cmap='YlGnBu')
#         axes[i + 1].set_title(f'Predicted Surface using {degree}th Order Polynomial Regression\nMSE: {mse:.2f}')
#         annotate_extrema(heatmap_predicted.values, ax, extrema_type='max')

#     heatmap_original = df.pivot(
#         index='latency', columns='scale', values='combo')
#     ax = sns.heatmap(heatmap_original, ax=axes[0], cmap="YlGnBu", annot=True)
#     axes[0].set_title('Combined Performance vs. Latency and Scale')
#     annotate_extrema(heatmap_original.values, ax, extrema_type='max')
#     plt.tight_layout()
#     plt.savefig(f'{data_folder}/TPcombo_poly_model_results.png')
#     plt.show()
