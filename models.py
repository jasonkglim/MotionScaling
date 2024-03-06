### Contains functions for different modeling approaches
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, RationalQuadratic
import numpy as np

# Object for model results, stores models of different metrics
class PerformanceModel:
	def __init__(self, train_inputs=None, train_output_dict=None):
		# TO DO: catch exception of bad args (given only inputs)
		if train_inputs == None:
			self.X = []
			self.y_dict = {}
		if train_inputs != None:
			self.X = train_inputs
			self.input_dim = self.X.shape[0]
			self.num_examples = self.X.shape[1]
			for metric, data in train_output_dict.items():
				self.y_dict[metric] = np.array(data).reshape((-1, 1))
		

	def train(self):
		pass
		




## Polynomail Regression
def PolyRegression(train_inputs, train_outputs, test_inputs, degree = 2):

	# Feature transformation
	poly = PolynomialFeatures(degree=degree)
	train_inputs_poly = poly.fit_transform(train_inputs)
	test_inputs_poly = poly.transform(test_inputs)

	# Train model on training set
	model = LinearRegression()
	model.fit(train_inputs_poly, train_outputs)

	# Predict over test inputs
	Y_pred = model.predict(test_inputs_poly)

	params = f"Coef: {model.coef_[1:]}, Intercept: {model.intercept_}"

	# Return predictions
	return Y_pred, params


## Gaussian Process Regression
def GPRegression(train_inputs, train_outputs, test_inputs, kernel):

	# Define the Gaussian Process kernel
	# kernel = ConstantKernel() * RBF() # Default RBF
	# kernel = ConstantKernel() * RationalQuadratic() # Rational Quadratic

	# Initialize the Gaussian Process Regressor with the chosen kernel
	gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)

	# Fit the model to the training data
	gp_model.fit(train_inputs, train_outputs)

	# Get mean and std of predictive distributions
	pred_mean, pred_std = gp_model.predict(test_inputs, return_std=True)

	return pred_mean, pred_std, gp_model.kernel_


## Bayesian Linear Regression
class BayesRegression(PerformanceModel):
	def __init__(self, train_inputs=None, train_output_dict=None, noise=1):
		super().__init__(train_inputs, train_output_dict)
		if len(self.X) > 0:
			self.prior_mean = np.zeros((self.input_dim, 1))
			self.prior_covar = np.identity(self.input_dim)
		self.noise = noise # Defines the variance of gaussian observation noise
		self.posterior_dict = {}
		self.prediction_dict = {}

	# Define custom prior for weights
	def set_prior(self, mean, var):
		if isinstance(mean, (int, float)):
			self.prior_mean = mean * np.ones((self.input_dim, 1))
		else:
			self.prior_mean = mean.reshape(-1, 1)
		if isinstance(var, (int, float)):
			self.prior_covar = np.identity(self.input_dim) * var
		else: 
			self.prior_covar = var

	def add_training_data(self, train_inputs, train_output_dict):
		# TO DO: catch exceptions for bad args, change to accept dictionary
		# if len(self.X) == 0: # if uninitialized
		self.X = train_inputs
		for metric, data in train_output_dict.items():
			self.y_dict[metric] = np.array(data).reshape((-1, 1))
		self.input_dim = self.X.shape[0]
		self.num_examples = self.X.shape[1]
		self.set_prior(0, 1e3) # TO DO change this!
		# else:
		# 	self.X = np.concatenate((self.X, train_inputs), 1)
		# 	for metric, data in train_output_dict.items():
		# 		data = np.array(data).reshape((-1, 1)) # convert to np column vector
		# 		self.y_dict[metric] = np.concatenate((self.y_dict[metric], data), 1)

	# Computes poseterior parameters from training data and prior 
	def train(self):
		for metric, y in self.y_dict.items():
			y = np.array(y).reshape((-1, 1))
			A = (self.X @ self.X.T / self.noise**2
				+ np.linalg.inv(self.prior_covar))
			posterior_covar = np.linalg.inv(A)
			# print(self.posterior_covar.shape)
			# print("X ", self.X.shape)
			# print("y ", self.y.shape)
			# print("prior covar ", self.prior_covar.shape)
			# print("prior mean ", self.prior_mean.shape)
			posterior_mean = (posterior_covar
								@ (self.X @ y / self.noise**2
									+ np.linalg.inv(self.prior_covar)
									@ self.prior_mean))
			self.posterior_dict[metric] = (posterior_mean, posterior_covar)

			# print("post mean ", self.posterior_mean.shape)
			# print("post covar ", self.posterior_covar.shape)
		return self.posterior_dict
	
	# Compute mean and covariance of predictive distribution
	def predict(self, test_input, prediction_df=None):
		'''
		Return mean and covar for predicted values over test_input
		args:
			test_input: array: each column represents an input, should be size input_dim x num_inputs
		'''
		for metric, (posterior_mean, posterior_covar) in self.posterior_dict.items():
			test_input = np.array(test_input)
			pred_mean = test_input.T @ posterior_mean
			pred_covar = test_input.T @ posterior_covar @ test_input
			self.prediction_dict[metric] = (pred_mean, pred_covar)
			if prediction_df is not None:
				prediction_df[metric] = pred_mean
				prediction_df[metric+"_var"] = np.diagonal(pred_covar)

		return self.prediction_dict
