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
		
	def max_unc_scale(self, prediction_df, metric):
		s =  prediction_df.loc[prediction_df.groupby('latency')[f"{metric}_var"].idxmax()]['scale'].values[0]
		return s, "max_uncertainty"
	
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
		
	
	
	
	
