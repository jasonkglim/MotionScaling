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
from models import BayesRegression
from scaling_policy import ScalingPolicy, BalancedScalingPolicy

def visualize_controller(obs_df, prediction_df):
	# Define ranges
	scale_range = [0.1, 0.15, 0.2, 0.4, 0.7, 1.0]
	latency_range = [0.25]

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
	fig, axes = plt.subplots(3, 2, figsize=(12, 6))

	# Throughput Heatmap
	sparse_throughput_heatmap = sparse_df.pivot(index="latency", columns="scale", values="throughput")
	sns.heatmap(sparse_throughput_heatmap, cmap='YlGnBu', ax=axes[0, 0], annot=True)
	axes[0, 0].set_title('Observed Throughput')
	axes[0, 0].set_xlabel('Scale')
	axes[0, 0].set_ylabel('Latency')

	# Total Error Heatmap
	sparse_error_heatmap = sparse_df.pivot(index="latency", columns="scale", values="total_error")
	sns.heatmap(sparse_error_heatmap, cmap='YlGnBu', ax=axes[0, 1], annot=True)
	axes[0, 1].set_title('Observed Total Error')
	axes[0, 1].set_xlabel('Scale')
	axes[0, 1].set_ylabel('Latency')

	# Predicted Heatmaps
	pred_throughput_heatmap = prediction_df.pivot(index="latency", columns="scale", values="throughput")
	sns.heatmap(pred_throughput_heatmap, cmap='YlGnBu', ax=axes[1, 0], annot=True)
	axes[1, 0].set_title('Predicted Mean Throughput')
	axes[1, 0].set_xlabel('Scale')
	axes[1, 0].set_ylabel('Latency')

	# Total Error Heatmap
	pred_error_heatmap = prediction_df.pivot(index="latency", columns="scale", values="total_error")
	sns.heatmap(pred_error_heatmap, cmap='YlGnBu', ax=axes[1, 1], annot=True)
	axes[1, 1].set_title('Predicted Mean Total Error')
	axes[1, 1].set_xlabel('Scale')
	axes[1, 1].set_ylabel('Latency')

	# Predicted Heatmaps
	pred_throughput_covar_heatmap = prediction_df.pivot(index="latency", columns="scale", values="throughput_var")
	sns.heatmap(pred_throughput_covar_heatmap, cmap='YlGnBu', ax=axes[2, 0], annot=True)
	axes[2, 0].set_title('Predicted Variance Throughput')
	axes[2, 0].set_xlabel('Scale')
	axes[2, 0].set_ylabel('Latency')

	# Total Error Heatmap
	pred_error_covar_heatmap = prediction_df.pivot(index="latency", columns="scale", values="total_error_var")
	sns.heatmap(pred_error_covar_heatmap, cmap='YlGnBu', ax=axes[2, 1], annot=True)
	axes[2, 1].set_title('Predicted Variance Total Error')
	axes[2, 1].set_xlabel('Scale')
	axes[2, 1].set_ylabel('Latency')

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
player = "user_sujaan"
player_df = all_datasets[player]
player_df = player_df[player_df["latency"] == 0.25]
metric_dict = {}

# Initialize model and control policy
scale_domain = [0.1, 0.15, 0.2, 0.4, 0.7, 1.0]
latency_domain = [0.25]
metric_list = ["throughput", "total_error"] # metrics to be tracked and modeled by PerformanceModel
obs_df = pd.DataFrame()
model = Bala()
policy = ScalingPolicy(scale_domain=scale_domain)
prediction_df = player_df[["latency", "scale"]].copy()

# Control loop:
full_latency_list = [l for l in latency_domain for _ in range(len(scale_domain))]
control_scale = 1.0
visited = []
for input_latency in full_latency_list:
	# while True:
	# 	control_scale = policy.random_scale()
	# 	if (input_latency, control_scale) not in visited or len(visited) == len(player_df):
	# 		break # TO DO change so that it breaks if

	# Select input_latency?
	obs_df = pd.concat([obs_df, player_df[(player_df["latency"] == input_latency) & (player_df["scale"] == control_scale)]])
	print(obs_df[["latency", "scale"]])
	# add obs_data to training data of PerformanceModel
	obs_input = obs_df[["latency", "scale"]].values.T # ensure input data is formatted as column vectors, d x N
	obs_output_dict = obs_df[metric_list].to_dict(orient='list')
	visited.append((input_latency, control_scale))
	# print(obs_df)
	# print(obs_input)
	# print(obs_output_dict)
	
	model.add_training_data(obs_input, obs_output_dict)
	model.train()
	test_input = player_df[["latency", "scale"]].values.T
	
	prediction_dict = model.predict(test_input, prediction_df)
	

	# # Input Performance Model to Scaling Policy
	# policy.update(model) 

	# # Visualize
	visualize_controller(obs_df, prediction_df)

	control_scale = policy.optimal_scale(prediction_df, metric="throughput")
	




	


