import numpy as np
from models import BayesRegression
import random

# Implements a scaling policy that chooses a control scale from performance model data
class ScalingPolicy:
	def __init__(self, scale_domain, policy_type=""):
		self.scale_domain = scale_domain
		self.policy_type = policy_type
		return
	
	def get_scale(self, prediction_df, metric, latency=None):
		if self.policy_type == "maxUnc":
			return self.max_unc_scale(prediction_df, metric, latency)
		if self.policy_type == "greedy":
			return self.optimal_scale(prediction_df, metric, latency)
		if self.policy_type == "random":
			return self.random_scale(prediction_df)
		print("Policy type not set!")
		return None, ""

	
	def random_scale(self, prediction_df  = None):
		if prediction_df is None:
			return random.choice([s for s in self.scale_domain]), "random"
		else:
			return random.choice([s for s in prediction_df["scale"].values]), "random"
		

	def max_unc_scale(self, prediction_df, metric, latency=None, level=1):
		# Ensure 'level' is at least 1 since we're dealing with ranks starting from 1
		level = max(level, 1)

		# Filter the DataFrame for the specified latency value
		if "latency" in prediction_df.columns:
			filtered_df = prediction_df[prediction_df['latency'] == latency]
		else:
			filtered_df = prediction_df.copy()

		# Find the nth largest value by metric_var within the filtered DataFrame
		# If there are fewer rows than 'level', handle appropriately
		if len(filtered_df) >= level:
			nth_largest_idx = filtered_df.nlargest(level, f"{metric}_var").index[-1]
			scale_value = filtered_df.loc[nth_largest_idx, 'scale']
			return scale_value, "max_uncertainty"
		else:
			# Handle the case where the filtered DataFrame has fewer than 'level' rows
			return np.nan, "error"
		

	def optimal_scale(self, prediction_df, metric, latency=None):
		if latency is not None:
			filtered_df = prediction_df[prediction_df['latency'] == latency]
		else:
			filtered_df = prediction_df.copy()
		optimal_scale = filtered_df.loc[filtered_df[metric].idxmax()]['scale']
		return optimal_scale, "optimal"


# Chooses greedily half the time, otw pick scale with highest uncertainty
class BalancedScalingPolicy(ScalingPolicy):
	def __init__(self, scale_domain):
		super().__init__(scale_domain)

	def get_scale(self, prediction_df, metric, latency=None):
		r = random.random()
		if r < 0.5:
			return self.optimal_scale(prediction_df, metric, latency)
		elif r > 0.5:
			return self.max_unc_scale(prediction_df, metric, latency)
		# else:
		# 	return self.random_scale(prediction_df)
		
	
	
	
# class StepSearch(ScalingPolicy):
# 	def __init__(self, scale_domain, prediction_df=None, use_metric=""):
# 		super().__init__(scale_domain)
# 		self.prediction_df = prediction_df
# 		self.use_metric = use_metric
		
# 	def get_scale(self, prediction_df=None, use_metric=""):
# 		if prediction_df is not None: # Update model if necessary
# 			self.prediction_df = prediction_df
# 		if use_metric: # Use different metric?
# 			self.use_metric = use_metric

		