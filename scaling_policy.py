import numpy as np
from models import BayesRegression
import random

# Implements a scaling policy that chooses a control scale from performance model data
class ScalingPolicy:
	def __init__(self, scale_domain):
		self.scale_domain = scale_domain
		return
	
	def random_scale(self):
		return random.choice(self.scale_domain)

	# adds new model data
	def update(self, model):
		pass

	# returns control scale
	def get_scale(self):
		pass