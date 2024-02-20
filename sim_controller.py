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
import itertools
# from models import BayesRegression
# from scaling_policy import ScalingPolicy

def visualize_controller(obs_df):
    # Define ranges
    scale_range = [0.1, 0.15, 0.3, 0.4, 0.7, 1.0]
    latency_range = [0.0, 0.25, 0.5, 0.75]

    # Create DataFrame for all combinations
    sparse_df = pd.DataFrame(list(itertools.product(latency_range, scale_range)), columns=["latency", "scale"])
    sparse_df['throughput'] = np.nan
    sparse_df['total_error'] = np.nan

    # Update sparse_df with data from obs_df
    obs_df_copy = obs_df.copy()
    sparse_df.set_index(['latency', 'scale'], inplace=True)
    obs_df_copy.set_index(['latency', 'scale'], inplace=True)
    sparse_df.update(obs_df_copy)
    sparse_df.reset_index(inplace=True)

    # Plotting
    fig, axes = plt.subplots(1, 2)

    # Throughput Heatmap
    sparse_throughput_heatmap = sparse_df.pivot("latency", "scale", "throughput")
    sns.heatmap(sparse_throughput_heatmap, cmap='YlGnBu', ax=axes[0], annot=True)
    axes[0].set_title('Observed Throughput')
    axes[0].set_xlabel('Scale')
    axes[0].set_ylabel('Latency')

    # Total Error Heatmap
    sparse_error_heatmap = sparse_df.pivot("latency", "scale", "total_error")
    sns.heatmap(sparse_error_heatmap, cmap='YlGnBu', ax=axes[1], annot=True)
    axes[1].set_title('Observed Total Error')
    axes[1].set_xlabel('Scale')
    axes[1].set_ylabel('Latency')

    plt.tight_layout()
    plt.show()



    # Plot observed data
    # fig, axes = plt.subplots(2, 3)
    

# Pattern to match the data files
file_pattern = "data_files/user_*/metric_df.csv"

# Initialize a dictionary to store one_user_one_user_dataframes for each dataset
all_datasets = {}

# Loop through each file that matches the file pattern
for filepath in glob.glob(file_pattern):
    # print(filepath)
    # print(filepath.split('/'))
    user_name = filepath.split('/')[1]
    # user_name = filepath.split('\\')[1]
    # print(f"Processing {filepath} dataset...")

    # Read in data file as a pandas dataframe
    data = pd.read_csv(filepath, index_col=0)

    # add weighted performance metric
    w = 1
    data["total_error"] = data['avg_osd'] + data['avg_target_error']
    data["weighted_performance"] = 10*data['throughput'] - w*data["total_error"]

    all_datasets[user_name] = data

# Combine datasets for Lizzie
lizzie1 = all_datasets["user_lizzie1"]
lizzie2 = all_datasets["user_lizzie2"]
combined_df = pd.concat([lizzie1, lizzie2])
all_datasets["user_lizzie"] = combined_df.groupby(['latency', 'scale']).mean().reset_index()

## Start modeling
player = "user_jason"
player_df = all_datasets[player]
metric_dict = {}

# Initialize model and control policy
# model = BayesRegression()
# policy = ScalingPolicy()
control_scales = [1.0, 0.1, 0.4, 0.7]
metric_list = ["throughput", "total_error"] # metrics to be tracked and modeled by PerformanceModel
obs_df = pd.DataFrame()

# Control loop:
input_latency = 0.0

for control_scale in control_scales:
    # Select input_latency?
    obs_df = pd.concat([obs_df, player_df[(player_df["latency"] == input_latency) & (player_df["scale"] == control_scale)]])
    print(obs_df[["latency", "scale"]])
    # add obs_data to training data of PerformanceModel
    obs_input = obs_df[["latency", "scale"]].values.T # ensure input data is formatted as column vectors, d x N
    obs_output_dict = obs_df[metric_list].to_dict(orient='list')
    # print(obs_df)
    # print(obs_input)
    # print(obs_output_dict)
    
    # model.add_training_data(obs_input, obs_output_dict)
    # model.train()
    # model_results = model.predict()

    # # Input Performance Model to Scaling Policy
    # policy.update(model)
    # control_scale = policy.get_scale()

    # # Visualize
    visualize_controller(obs_df)
    




    


