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


def plot_full_data_set(groups):

	fig, axes = plt.subplots(1, 2, figsize=(12, 6))
	fig.suptitle("Full Data")
	scale = []
	mean_tp = []
	std_tp = []
	mean_err = []
	std_err = []

	for name, data in groups:
		scale.append(data["scale"].values[0])
		mean_tp.append(data["throughput"].mean())
		std_tp.append(data["throughput"].std())
		mean_err.append(data["total_error"].mean())
		std_err.append(data["total_error"].std())

	mean_tp = np.array(mean_tp)
	std_tp = np.array(std_tp)
	mean_err = np.array(mean_err)
	std_err = np.array(std_err)

	axes[0].fill_between(scale, mean_tp-std_tp, mean_tp+std_tp)
	axes[0].plot(scale, mean_tp, linestyle='--', marker='o', color='black')
	axes[1].fill_between(scale, mean_err-std_err, mean_err+std_err)
	axes[1].plot(scale, mean_err, linestyle='--', marker='o', color='black')

	for name, data in groups:
		axes[0].scatter(data["scale"], data["throughput"], color='red', marker='x')
		axes[1].scatter(data["scale"], data["total_error"], color='red', marker='x')

	axes[0].set_xlabel("Scale")
	axes[0].set_ylabel("Throughput")
	axes[1].set_xlabel("Scale")
	axes[1].set_ylabel("Total Error")

	plt.savefig("data_files/user_jason_new/full_data.png")
	plt.show()

	# Plot observed data
	# fig, axes = plt.subplots(2, 3)

def plot_2d(ref_df, obs_data, prediction_df, iteration, control_scale, policy_choice, save_data_folder, trial):
	
	plt.figure()
	plt.title(f"Control Scale Chosen: {control_scale} by {policy_choice}")
	
	plt.scatter(obs_data["scale"], obs_data["throughput"], color='blue')
	plt.scatter(ref_df["scale"], ref_df["throughput"], marker='x', color='green')
	plt.plot(prediction_df["scale"], prediction_df["throughput"], color='black', linestyle='--')
	plt.fill_between(prediction_df["scale"], 
					  prediction_df["throughput"]-prediction_df["throughput_var"],
					  prediction_df["throughput"]+prediction_df["throughput_var"],
					  alpha=0.3)
	
	# axes[1].scatter(obs_data["scale"], obs_df["total_error"])
	# axes[1].plot(prediction_df["scale"], prediction_df["total_error"], color='black', linestyle='--')
	# axes[1].fill_between(prediction_df["scale"], 
	# 				  prediction_df["total_error"]-prediction_df["total_error_var"],
	# 				  prediction_df["total_error"]+prediction_df["total_error_var"],
	# 				  alpha=0.3)
	
	plt.savefig(f"{save_data_folder}/trial{trial}_iter{iteration}.png")
	plt.close()
	
	

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

# Cherry picking good values to use as "ground truth", to sample training points about and to act as test set
ref_data = [0.95, 1.25, 1.4, 1.56, 1.7, 1.8, 1.6, 1.4]
scale_domain = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
ref_df = pd.DataFrame({
	'scale': scale_domain,
	'throughput': ref_data
})
obs_var = 0.2 # assumed variance for sampling training points
true_optimal_scale_throughput = 0.6 
metric_list = ["throughput"] # metrics to be tracked and modeled by PerformanceModel
test_input = scale_domain #np.linspace(0, 1, 100)

# Initialize Scaling policy
policy_type = "maxUnc"
policy = ScalingPolicy(scale_domain=scale_domain, policy_type=policy_type)

max_n_train = 10 # max number of training points for each simulated trial
num_trials = 100

# Simulate multiple trials
for trial in range(num_trials):
	print("Trial ", trial)
	# Save file paths
	save_data_folder = f"controller_data_files/neweval/{policy_type}"
	os.makedirs(save_data_folder, exist_ok=True)

	# Init variables for evaluating trial
	# metric_dict = {}
	mse_scores = {"throughput": [], "total_error": []}
	optimal_scale_errors = {"throughput": [], "total_error": []}
	obs_metric_avgs = {"throughput": [], "total_error": []}
	obs_metric_optimums = {"throughput": [], "total_error": []}

	# Initialize data vars, models, and scaling policy for this trial
	obs_data = {"scale": [], "throughput": []}
	obs_df = pd.DataFrame()
	model = BayesRegression()
	# model.homogenize(True)
	model.set_poly_transform(degree=2)
	
	prediction_df_dense = pd.DataFrame({
		# "latency": test_input[0,:],
		"scale": test_input
	})
	prediction_df = pd.DataFrame({
		"scale": np.array(scale_domain)
	})

	control_scale = 0.4
	policy_choice = "init"
	for i in range(max_n_train):
		print("Iteration ", i)

		# Sample training point
		mean_obs = ref_df[ref_df["scale"] == control_scale]["throughput"]
		sample_obs = np.random.normal(loc=mean_obs, scale=obs_var)
		obs_data["scale"].append(control_scale)
		obs_data["throughput"].append(sample_obs)

		model.add_training_data(train_inputs = control_scale, train_output_dict={"throughput": sample_obs})
		model.train()
		
		prediction_dict = model.predict(test_input, prediction_df)

		# Compute eval metrics
		mse_scores_throughput = mean_squared_error(prediction_df["throughput"], ref_df["throughput"])
		# mse_scores_error = mean_squared_error(prediction_df["total_error"], ref_df["total_error"])
		pred_optimal_scale_throughput = prediction_df.loc[prediction_df["throughput"].idxmax()]['scale']
		# pred_optimal_scale_error = prediction_df.loc[prediction_df["total_error"].idxmax()]['scale']
		# optimal_scale_error_throughput = mean_squared_error(true_optimal_scale_throughput, pred_optimal_scale_throughput)
		# optimal_scale_error_error = mean_squared_error(true_optimal_scale_error, pred_optimal_scale_error)
		optimal_scale_error_throughput = np.square(true_optimal_scale_throughput - pred_optimal_scale_throughput)
		# optimal_scale_error_error = np.square(true_optimal_scale_error - pred_optimal_scale_error)
		mse_scores["throughput"].append(mse_scores_throughput)
		# mse_scores["total_error"].append(mse_scores_error)
		optimal_scale_errors["throughput"].append(optimal_scale_error_throughput)
		# optimal_scale_errors["total_error"].append(optimal_scale_error_error)
		avg_throughput = np.mean(obs_data["throughput"])
		max_throughput = np.max(obs_data["throughput"])
		obs_metric_avgs["throughput"].append(avg_throughput)
		obs_metric_optimums["throughput"].append(max_throughput)


		# # Visualize
		# visualize_controller(obs_df, prediction_df, i, control_scale, policy_choice, save_data_folder)
		if trial % 10 == 0:
			plot_2d(ref_df, obs_data, prediction_df, i, control_scale, policy_choice, save_data_folder, trial)
		# 

		# Get next control_scale
		control_scale, policy_choice = policy.get_scale(prediction_df, metric="throughput")
		
	### Save evaluation metrics
	eval_data = dict(
		mse_scores = mse_scores,
		optimal_scale_errors = optimal_scale_errors,
		obs_metric_avgs = obs_metric_avgs,
		obs_metric_optimums = obs_metric_optimums
	)

	with open(f"{save_data_folder}/trial{trial}_eval_data.pkl", "wb") as f:
		pickle.dump(eval_data, f, protocol=pickle.HIGHEST_PROTOCOL)


