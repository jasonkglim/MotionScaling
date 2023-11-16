### script for trying models
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import seaborn as sns

def annotate_extrema(data, ax, extrema_type='max'):
    if extrema_type == 'max':
        extrema_index = np.argmax(data, axis=1)
        color = 'red'
    if extrema_type == 'min':
        extrema_index = np.argmin(data, axis=1)
        color = 'orange'
    for i, max_col in enumerate(extrema_index):
        ax.add_patch(plt.Rectangle((max_col, i), 1, 1, fill=False, edgecolor=color, lw=3))

for set_num in range(2, 7):
    # Read in data
#    set_num = 6
    data_folder = f"data_files/set{set_num}"
    df = pd.read_csv(f'{data_folder}/metric_df.csv')

    latency_vals = [0.0, 0.2, 0.4, 0.6, 0.8]
    scale_vals = [0.4, 0.6, 0.8, 1.0, 1.2]

    # Extracting the independent variables and the dependent variable
    X = df[['latency', 'scale']]
    df['combo'] = 10*df['throughput'] - df['avg_osd'] - df['avg_target_error']
    y = df['combo']

    # Creating polynomial features
    degrees = [2, 3, 4]  # List of degrees for different models

    fig, axes = plt.subplots(1, len(degrees) + 1, figsize=(5 * (len(degrees) + 1), 6))  # Adjusting subplots

    for i, degree in enumerate(degrees):
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)

        # Fitting the polynomial features to Linear Regression
        model = LinearRegression()
        model.fit(X_poly, y)

        # Calculating Mean Squared Error
        y_pred = model.predict(X_poly)
        mse = mean_squared_error(y, y_pred)

        # Plot surface predicted by model
        mesh_df = pd.DataFrame([[i, j] for i in latency_vals for j in scale_vals], columns=['latency', 'scale'])
        X_mesh = mesh_df[['latency', 'scale']]
        X_mesh_poly = poly.fit_transform(X_mesh)

        # Predicting the values for the meshgrid
        predicted_values = model.predict(X_mesh_poly)

        mesh_df[f'predicted_{degree}'] = predicted_values

        # Plotting the heatmaps
        heatmap_predicted = mesh_df.pivot(
            index='latency', columns='scale', values=f'predicted_{degree}')
        ax = sns.heatmap(heatmap_predicted, ax=axes[i + 1], annot=True, cmap='YlGnBu')
        axes[i + 1].set_title(f'Predicted Surface using {degree}th Order Polynomial Regression\nMSE: {mse:.2f}')
        annotate_extrema(heatmap_predicted.values, ax, extrema_type='max')

    heatmap_original = df.pivot(
        index='latency', columns='scale', values='combo')
    ax = sns.heatmap(heatmap_original, ax=axes[0], cmap="YlGnBu", annot=True)
    axes[0].set_title('Combined Performance vs. Latency and Scale')
    annotate_extrema(heatmap_original.values, ax, extrema_type='max')
    plt.tight_layout()
    plt.savefig(f'{data_folder}/TPcombo_poly_model_results.png')
    plt.show()
