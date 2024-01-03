import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
        
        # Add a rectangle around the corresponding cell in the heatmap
        ax.add_patch(plt.Rectangle((col_idx, row_idx), 1, 1, fill=False, edgecolor=color, lw=3))