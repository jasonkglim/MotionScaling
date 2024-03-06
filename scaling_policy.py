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
			return random.choice(self.scale_domain)
		else:
			return random.choice([s for s in self.scale_domain if s not in visited])

	def optimal_scale(self, metric):
			pass

	# adds new model data
	def update(self, model):
		pass

	# returns control scale
	def get_scale(self):
		pass


# Chooses greedily half the time, otw pick scale with highest uncertainty
class BalancedScalingPolicy(ScalingPolicy):
	def __init__(self, scale_domain):
		super().__init__(scale_domain)

	def get_scale(self, prediction_df):
		if random.random() < 0.5:
			return self.optimal_scale(prediction_df)
		else:
			return self.max_unc_scale(prediction_df)
		
	

