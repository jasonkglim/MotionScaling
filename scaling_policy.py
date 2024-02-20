import numpy as np
from models import BayesRegression

# Implements a scaling policy that chooses a control scale from performance model data
class ScalingPolicy:
	def __init__(self):
		pass

	# adds new model data
	def update(self, model):
		pass

	# returns control scale
	def get_scale(self):
		pass