import numpy as np
from models import BayesRegression
import random

# Implements a scaling policy that chooses a control scale from performance model data
class ScalingPolicy:
	def __init__(self, scale_domain):
		self.scale_domain = scale_domain
		return
	
	def random_scale(self, visited=None):
		# TO DO fix this
		if visited is None:
			return random.choice(self.scale_domain), "random"
		else:
			return random.choice([s for s in self.scale_domain if s not in visited]), "random"
		
	def max_unc_scale(self, prediction_df, metric, latency, level=1):
		# Ensure 'level' is at least 1 since we're dealing with ranks starting from 1
		level = max(level, 1)

		# Filter the DataFrame for the specified latency value
		filtered_df = prediction_df[prediction_df['latency'] == latency]

		# Find the nth largest value by metric_var within the filtered DataFrame
		# If there are fewer rows than 'level', handle appropriately
		if len(filtered_df) >= level:
			nth_largest_idx = filtered_df.nlargest(level, f"{metric}_var").index[-1]
			scale_value = filtered_df.loc[nth_largest_idx, 'scale']
			return scale_value, "max_uncertainty"
		else:
			# Handle the case where the filtered DataFrame has fewer than 'level' rows
			return np.nan, "error"
		

	
	def optimal_scale(self, prediction_df, metric):
		optimal_scale = prediction_df.loc[prediction_df.groupby('latency')[metric].idxmax()]['scale'].values[0]
		return optimal_scale, "optimal"


# Chooses greedily half the time, otw pick scale with highest uncertainty
class BalancedScalingPolicy(ScalingPolicy):
	def __init__(self, scale_domain):
		super().__init__(scale_domain)

	def get_scale(self, prediction_df, metric):
		r = random.random()
		if r < 0.2:
			return self.optimal_scale(prediction_df, metric)
		elif r > 0.8:
			return self.max_unc_scale(prediction_df, metric)
		else:
			return self.random_scale()
		
	
	
	
	
