import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Stratified sampling for smarter training set split
def stratified_sample(data, n_train):
    grouped = data.groupby('latency')
    num_col_latency = len(grouped)
    k = n_train // num_col_latency

    train_samples = pd.DataFrame()
    for name, group in grouped:
        sampled_group = group.sample(n=min(k, len(group)))
        train_samples = pd.concat([train_samples, sampled_group])
        n_train -= len(sampled_group)
    
    remaining_points = data[~data.index.isin(train_samples.index)]
    if n_train > 0 and not remaining_points.empty:
        additional_samples = remaining_points.sample(n=n_train, random_state=42)
        train_samples = pd.concat([train_samples, additional_samples])
    
    test_samples = data[~data.index.isin(train_samples.index)]
    # print(train_samples)
    return train_samples, test_samples

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

        circle_center_x = col_idx + 0.5  # X coordinate of the circle's center
        circle_center_y = row_idx + 0.5  # Y coordinate of the circle's center
        radius = 0.4  # Set the radius of the circle

        circle = plt.Circle((circle_center_x, circle_center_y), radius, fill=False, edgecolor=color, lw=3)
        ax.add_patch(circle)
        
        # # Add a rectangle around the corresponding cell in the heatmap
        # ax.add_patch(plt.Rectangle((col_idx, row_idx), 1, 1, fill=False, edgecolor=color, lw=3))


# Picks training points evenly distributed across input domain
def even_train_split(data, n_train):
    # Ensure that n_train is within the valid range
    if not 1 <= n_train <= len(data) - 2:
        raise ValueError("n_train must be between 1 and n-2")

    # Sort data based on 'latency' and 'scale' to ensure even distribution
    data_sorted = data.sort_values(by=['latency', 'scale'])

    # Select training data points evenly
    indices = np.round(np.linspace(0, len(data_sorted) - 1, n_train)).astype(int)
    train_data = data_sorted.iloc[indices]

    # The rest of the data will be used as the test set
    test_data = data_sorted.drop(train_data.index)

    return train_data, test_data


# Function to add red border to maximum value in each row
def annotate_extrema(data, ax, extrema_type='max'):
    if extrema_type == 'max':
        extrema_index = np.nanargmax(data, axis=1)
        color = 'red'
    if extrema_type == 'min':
        extrema_index = np.nanargmin(data, axis=1)
        color = 'orange'
    for i, max_col in enumerate(extrema_index):
        ax.add_patch(plt.Rectangle((max_col, i), 1, 1, fill=False, edgecolor=color, lw=3))

# Visualize model results by plotting heatmaps for original data and predictions
def model_heatmaps(data, dense_df, X_train, user, metric, model_type=""):
    fig, ax = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle(f"{model_type} Model Results for {user} for {metric} metric, using {len(X_train)} training points")
    # Original data heatmap with all points highlighted (now all are training points)
    original_data = data.pivot(
        index='latency', columns='scale', values=metric
    )
    sns.heatmap(original_data, cmap='YlGnBu', ax=ax[0], annot=True)
    annotate(ax[0], original_data, X_train, color='green')
    ax[0].set_title('Original Data')
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
    sns.heatmap(dense_pred_data, cmap='YlGnBu', ax=ax[2])
    # annotate(ax[1], dense_pred_data, X_train, color='green')
    ax[2].set_title('Predicted Data over Dense Input')
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
    folder = f"data_files/{user}/model_heatmaps/{metric}/{model_type}"
    if not os.path.exists(f"data_files/{user}/model_heatmaps"):
        os.mkdir(f"data_files/{user}/model_heatmaps")
    if not os.path.exists(f"data_files/{user}/model_heatmaps/{metric}"):
        os.mkdir(f"data_files/{user}/model_heatmaps/{metric}")
    if not os.path.exists(f"data_files/{user}/model_heatmaps/{metric}/{model_type}"):
        os.mkdir(f"data_files/{user}/model_heatmaps/{metric}/{model_type}")
    filepath = f"data_files/{user}/model_heatmaps/{metric}/{model_type}/ntrain_{len(X_train)}.png"
    plt.savefig(filepath)
    plt.show()
    plt.close()