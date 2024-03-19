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
from utils import annotate_extrema
import os
import pickle

def visualize_controller(obs_df, prediction_df, iteration, control_scale, policy_choice, save_data_folder):
	# Define ranges
	# scale_domain = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
	# latency_domain = [0.25]
	scale_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] #[0.1, 0.15, 0.2, 0.4, 0.7, 1.0]
	latency_range = [0.25]

	# Create DataFrame for all combinations
	sparse_df = pd.DataFrame(list(itertools.product(latency_range, scale_range)), columns=["latency", "scale"])
	sparse_df['throughput'] = np.nan
	sparse_df['total_error'] = np.nan

	# Update sparse_df with data from obs_df
	obs_df_copy = obs_df.copy()
	obs_df_copy = obs_df_copy.groupby(['latency', 'scale']).mean().reset_index()
	sparse_df.set_index(['latency', 'scale'], inplace=True)
	obs_df_copy.set_index(['latency', 'scale'], inplace=True)
	sparse_df.update(obs_df_copy)
	sparse_df.reset_index(inplace=True)

	# Plotting
	fig, axes = plt.subplots(3, 2, figsize=(12, 6))
	fig.suptitle(f"Control Scale Chosen: {control_scale} by {policy_choice}")

	# Throughput Heatmap
	sparse_throughput_heatmap = sparse_df.pivot(index="latency", columns="scale", values="throughput")
	sns.heatmap(sparse_throughput_heatmap, cmap='YlGnBu', ax=axes[0, 0], annot=True, fmt="0.3g")
	axes[0, 0].set_title('Observed Throughput')
	axes[0, 0].set_xlabel('Scale')
	axes[0, 0].set_ylabel('Latency')
	annotate_extrema(sparse_throughput_heatmap.values, axes[0, 0])
	annotate_extrema(sparse_throughput_heatmap.values, axes[0, 0], "min")

	# Total Error Heatmap
	sparse_error_heatmap = sparse_df.pivot(index="latency", columns="scale", values="total_error")
	sns.heatmap(sparse_error_heatmap, cmap='YlGnBu', ax=axes[0, 1], annot=True, fmt="0.3g")
	axes[0, 1].set_title('Observed Total Error')
	axes[0, 1].set_xlabel('Scale')
	axes[0, 1].set_ylabel('Latency')
	annotate_extrema(sparse_error_heatmap.values, axes[0, 1], "min")
	annotate_extrema(sparse_error_heatmap.values, axes[0, 1], "max")


	# Predicted Heatmaps
	pred_throughput_heatmap = prediction_df.pivot(index="latency", columns="scale", values="throughput")
	sns.heatmap(pred_throughput_heatmap, cmap='YlGnBu', ax=axes[1, 0], annot=True, fmt="0.3g")
	axes[1, 0].set_title('Predicted Mean Throughput')
	axes[1, 0].set_xlabel('Scale')
	axes[1, 0].set_ylabel('Latency')
	annotate_extrema(pred_throughput_heatmap.values, axes[1, 0])
	annotate_extrema(pred_throughput_heatmap.values, axes[1, 0], "min")


	# Total Error Heatmap
	pred_error_heatmap = prediction_df.pivot(index="latency", columns="scale", values="total_error")
	sns.heatmap(pred_error_heatmap, cmap='YlGnBu', ax=axes[1, 1], annot=True, fmt="0.3g")
	axes[1, 1].set_title('Predicted Mean Total Error')
	axes[1, 1].set_xlabel('Scale')
	axes[1, 1].set_ylabel('Latency')
	annotate_extrema(pred_error_heatmap.values, axes[1, 1], "min")
	annotate_extrema(pred_error_heatmap.values, axes[1, 1], "max")


	# Predicted Heatmaps
	pred_throughput_covar_heatmap = prediction_df.pivot(index="latency", columns="scale", values="throughput_var")
	sns.heatmap(pred_throughput_covar_heatmap, cmap='YlGnBu', ax=axes[2, 0], annot=True, fmt="0.3g")
	axes[2, 0].set_title('Predicted Variance Throughput')
	axes[2, 0].set_xlabel('Scale')
	axes[2, 0].set_ylabel('Latency')
	annotate_extrema(pred_throughput_covar_heatmap.values, axes[2, 0], "min")
	annotate_extrema(pred_throughput_covar_heatmap.values, axes[2, 0], "max")


	# Total Error Heatmap
	pred_error_covar_heatmap = prediction_df.pivot(index="latency", columns="scale", values="total_error_var")
	sns.heatmap(pred_error_covar_heatmap, cmap='YlGnBu', ax=axes[2, 1], annot=True, fmt="0.3g")
	axes[2, 1].set_title('Predicted Variance Total Error')
	axes[2, 1].set_xlabel('Scale')
	axes[2, 1].set_ylabel('Latency')
	annotate_extrema(pred_error_covar_heatmap.values, axes[2, 1], "min")
	annotate_extrema(pred_error_covar_heatmap.values, axes[2, 1], "max")


	plt.tight_layout()
	plt.savefig(f"{save_data_folder}/{iteration}.png")
	plt.close()



	# Plot observed data
	# fig, axes = plt.subplots(2, 3)
	

# # Pattern to match the data files
# file_pattern = "data_files/user_*/metric_df.csv"

# # Initialize a dictionary to store one_user_one_user_dataframes for each dataset
# all_datasets = {}

# # Loop through each file that matches the file pattern
# for filepath in glob.glob(file_pattern):
# 	# print(filepath)
# 	# print(filepath.split('/'))
# 	user_name = filepath.split('/')[1]
# 	# user_name = filepath.split('\\')[1]
# 	# print(f"Processing {filepath} dataset...")

# 	# Read in data file as a pandas dataframe
# 	data = pd.read_csv(filepath, index_col=0)

# 	# add weighted performance metric
# 	w = 1
# 	data["total_error"] = data['avg_osd'] + data['avg_target_error']
# 	data["weighted_performance"] = 10*data['throughput'] - w*data["total_error"]

# 	all_datasets[user_name] = data

# # Combine datasets for Lizzie
# lizzie1 = all_datasets["user_lizzie1"]
# lizzie2 = all_datasets["user_lizzie2"]
# combined_df = pd.concat([lizzie1, lizzie2])
# all_datasets["user_lizzie"] = combined_df.groupby(['latency', 'scale']).mean().reset_index()

## Start modeling
# player = "user_jason_new"
# player_df = all_datasets[player]
# player_df = player_df[player_df["latency"] == 0.25]

save_data_folder = "controller_data_files/50-50_greedy_max_unc"
os.makedirs(save_data_folder, exist_ok=True)
player_df_full = pd.read_csv("data_files/user_jason_new/obs_metric_data.csv", index_col=0)
player_df_avg = player_df_full.groupby(["latency", "scale"]).mean().reset_index()
true_optimal_scale_throughput = player_df_avg.loc[player_df_avg.groupby('latency')["throughput"].idxmax()]['scale'].values[0]
true_optimal_scale_error = player_df_avg.loc[player_df_avg.groupby('latency')["total_error"].idxmax()]['scale'].values[0]
player_df = player_df_full.copy()
metric_dict = {}
mse_scores = {"throughput": [], "total_error": []}
optimal_scale_errors = {"throughput": [], "total_error": []}

# Initialize model and control policy
scale_domain = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
latency_domain = [0.25]
metric_list = ["throughput", "total_error"] # metrics to be tracked and modeled by PerformanceModel
obs_df = pd.DataFrame()
model = BayesRegression()
model.set_poly_transform(degree=2)
policy = BalancedScalingPolicy(scale_domain=scale_domain)
prediction_df = player_df[["latency", "scale"]].copy()
test_input = np.array([[l, s] for l in latency_domain for s in scale_domain]).T
prediction_df = pd.DataFrame({
	"latency": test_input[0,:],
	"scale": test_input[1,:]
})

# Control loop:
full_latency_list = [l for l in latency_domain for _ in range(len(scale_domain))]
control_scale, policy_choice = policy.random_scale()
# policy_choice = "random"
visited = []
input_latency = 0.25
for i in range(38): #, input_latency in enumerate(full_latency_list):
	print("Iteration ", i)
	# while True:
	# 	control_scale = policy.random_scale()
	# 	if (input_latency, control_scale) not in visited or len(visited) == len(player_df):
	# 		break # TO DO change so that it breaks if

	# Select input_latency?
	filtered_df = player_df[(player_df["latency"] == input_latency) & (player_df["scale"] == control_scale)]
	# current_obs_df = player_df[(player_df["latency"] == input_latency) & (player_df["scale"] == control_scale)].iloc[0]
	current_obs_df = filtered_df.iloc[0:1]
	print(current_obs_df)
	player_df.drop(current_obs_df.index, inplace=True)
	obs_df = pd.concat([obs_df, current_obs_df])
	# print(obs_df[["latency", "scale"]])
	# add obs_data to training data of PerformanceModel
	obs_input = current_obs_df[["latency", "scale"]].values.T # ensure input data is formatted as column vectors, d x N
	obs_output_dict = current_obs_df[metric_list].to_dict(orient='list')
	visited.append((input_latency, control_scale))
	# print(obs_df)
	# print(obs_input)
	# print(obs_output_dict)
	
	model.add_training_data(obs_input, obs_output_dict)
	model.train()
	
	prediction_dict = model.predict(test_input, prediction_df)

	# Compute eval metrics
	test_set = player_df.groupby(["latency", "scale"]).mean().reset_index()
	prediction_df_test = pd.merge(prediction_df, test_set["scale"], on="scale", how="inner")
	mse_scores_throughput = mean_squared_error(prediction_df_test["throughput"], test_set["throughput"])
	mse_scores_error = mean_squared_error(prediction_df_test["total_error"], test_set["total_error"])
	pred_optimal_scale_throughput = prediction_df.loc[prediction_df.groupby('latency')["throughput"].idxmax()]['scale'].values[0]
	pred_optimal_scale_error = prediction_df.loc[prediction_df.groupby('latency')["total_error"].idxmax()]['scale'].values[0]
	# optimal_scale_error_throughput = mean_squared_error(true_optimal_scale_throughput, pred_optimal_scale_throughput)
	# optimal_scale_error_error = mean_squared_error(true_optimal_scale_error, pred_optimal_scale_error)
	optimal_scale_error_throughput = np.square(true_optimal_scale_throughput - pred_optimal_scale_throughput)
	optimal_scale_error_error = np.square(true_optimal_scale_error - pred_optimal_scale_error)
	mse_scores["throughput"].append(mse_scores_throughput)
	mse_scores["total_error"].append(mse_scores_error)
	optimal_scale_errors["throughput"].append(optimal_scale_error_throughput)
	optimal_scale_errors["total_error"].append(optimal_scale_error_error)

	# # Visualize
	visualize_controller(obs_df, prediction_df, i, control_scale, policy_choice, save_data_folder)

	found_scale = False
	control_scale, policy_choice = policy.get_scale(prediction_df_test, latency=input_latency, metric="throughput")
	# If chosen control_scale doesn't have any training points left, pick next most uncertain
	level = 2 
	# while not found_scale:
	if control_scale not in player_df["scale"].value_counts():
		found_scale = True
		# else:
		# 	control_scale, policy_choice = policy.max_unc_scale(prediction_df, metric="throughput", latency=input_latency, level=level)
		# 	level += 1
	


### Save evaluation metrics
eval_data = dict(
	mse_scores = mse_scores,
	optimal_scale_errors = optimal_scale_errors
)

with open(f"{save_data_folder}/eval_data.pkl", "wb") as f:
	pickle.dump(eval_data, f, protocol=pickle.HIGHEST_PROTOCOL)


