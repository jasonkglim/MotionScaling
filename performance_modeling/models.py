### Contains functions for different modeling approaches
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, RationalQuadratic
import numpy as np
import matplotlib.pyplot as plt

# Object for model results, stores models of different metrics
class PerformanceModel:
	def __init__(self, train_inputs=None, train_output_dict=None): 
		# TO DO: catch exceptions of bad args (given only inputs, etc.)
		'''
		Base Class for modeling performance metrics
		args:
			- train_input: base class assumes row vector format, N x d
			- train_output_dict: dictionary of output values for each metric
		'''
		if train_inputs is None:
			train_inputs = np.array([])
		if len(train_inputs) == 0: # If not initialized with any training data..
			self.X = []
			self.y_dict = {}
			self.input_dim = 0
			self.num_examples = 0
		else:			
			self.X = np.array(train_inputs)
			if len(self.X.shape) <= 1: # Assumes 1D array means set of 1D examples
				self.X = self.X.reshape(-1, 1) 
			self.num_examples = self.X.shape[0]
			self.input_dim = self.X.shape[1]
			self.y_dict = {}
			for metric, data in train_output_dict.items():
				data = np.array(data)
				# print(data.shape)
				if data.shape[0] != self.num_examples:
					print("Number of input/output examples don't match!!")
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


# ## Gaussian Process Regression
# def GPRegression(train_inputs, train_outputs, test_inputs, kernel):

# 	# Define the Gaussian Process kernel
# 	# kernel = ConstantKernel() * RBF() # Default RBF
# 	# kernel = ConstantKernel() * RationalQuadratic() # Rational Quadratic

# 	# Initialize the Gaussian Process Regressor with the chosen kernel
# 	gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)

# 	# Fit the model to the training data
# 	gp_model.fit(train_inputs, train_outputs)

# 	# Get mean and std of predictive distributions
# 	pred_mean, pred_std = gp_model.predict(test_inputs, return_std=True)

# 	return pred_mean, pred_std, gp_model.kernel_
class GPRegression(PerformanceModel):
	def __init__(self, train_inputs=[], train_output_dict=[], kernel=""):
		super().__init__(train_inputs, train_output_dict)
		self.prediction_dict = {}
		self.kernel = kernel
		self.kernel_params = {}
		self.test_input = []
		# if self.kernel:
		# 	self.gp_model = GaussianProcessRegressor(kernel=self.kernel,
		# 											n_restarts_optimizer=10,
		# 											random_state=42)

	def add_training_data(self, train_inputs=[], train_output_dict=[]):
		# TO DO: catch exceptions for bad args
		if len(self.X) == 0: # if uninitialized
			self.X = np.array(train_inputs)
			if len(self.X.shape) <= 1:
				self.X = self.X.reshape(-1, 1) # Assumes a 1D array represents list of 1D examples
			self.num_examples = self.X.shape[0]
			self.input_dim = self.X.shape[1]
			# if self.transform:
			# 	self.X_trans = self.poly.fit_transform(train_inputs.T).T
			# 	self.input_dim = len(self.X_trans)
			# if self.flag_homogenize:
			# 	self.homogenize()
			# self.prior_mean = np.zeros((self.input_dim, 1))
			# self.prior_covar = np.identity(self.input_dim) * 1e3
			# self.set_prior(self.prior_mean, self.prior_covar)
			for metric, data in train_output_dict.items():
				self.y_dict[metric] = np.array(data).reshape(-1, 1)
		else: # Already have some data
			train_inputs = np.array(train_inputs)
			if len(train_inputs.shape) <= 1:
				train_inputs = train_inputs.reshape(-1, 1)
			self.X = np.concatenate((self.X, train_inputs), 0)
			self.num_examples = self.X.shape[0]
			# if self.transform:
			# 	self.X_trans = np.concatenate((self.X_trans, self.poly.transform(train_inputs.T).T), 0)
			for metric, data in train_output_dict.items():
				data = np.array(data).reshape((-1, 1)) # convert to np column vector
				self.y_dict[metric] = np.concatenate((self.y_dict[metric], data), 0)

	def train_predict(self, test_input=[], prediction_df=None):
		for metric, data in self.y_dict.items():
			gp_model = GaussianProcessRegressor(kernel=self.kernel,
													n_restarts_optimizer=10,
													random_state=42)
			gp_model.fit(self.X, data)
			if len(self.test_input) > 0: # If self.test_input has been set, it overrides
				if len(test_input) > 0:  
					print("Self.test_input already set, using it instead.")
				pred_mean, pred_std = gp_model.predict(self.test_input, return_std=True)
			elif len(test_input) > 0 and len(self.test_input == 0):
				pred_mean, pred_std = gp_model.predict(test_input, return_std=True)
			else:
				print("No test input available")

			self.kernel_params[metric] = gp_model.kernel_
			self.prediction_dict[metric] = (pred_mean, pred_std)
			if prediction_df is not None:
				prediction_df[metric] = pred_mean
				prediction_df[metric] = pred_std
			
		return self.prediction_dict, self.kernel_params
	
	def set_test_input(self, test_input):
		self.test_input = np.array(test_input)
		if len(self.test_input.shape) <= 1:
			self.test_input = self.test_input.reshape(-1, 1)
		# elif self.test_input.shape[1] != self.X.shape[1]:
		# 	print("dimension error")

	# def visualize(self):
	# 	# by default use self.test_input
	# 	self.train_predict()

	# 	fig, ax = plt.subplots(1, len(self.y_dict.keys()), )
	# 	for metric, data, i in enumerate(self.y_dict.items()):
	# 		ax[i].plot(self.X, data, marker='o')
	# 		ax[i].plot(self.test_input, self.prediction_dict[metric][0])

## Bayesian Linear Regression
class BayesRegression(PerformanceModel):
	'''
	TODO: fix/test self.transform function
	'''
	def __init__(self, train_inputs=None, train_output_dict=None, obs_noise_std=1, prior_base_var=1):
		super().__init__(train_inputs, train_output_dict)
		# Initialize prior mean, default is 0 and identity
		# if len(prior_mean) == 0:
		# 	self.set_prior(0, )
		self.homogenize = False
		self.transform = False
		self.obs_noise_std = obs_noise_std # Defines the variance of gaussian observation noise
		self.prior_base_var = prior_base_var # Defines diagonal elements of prior covariance
		self.has_informed_prior = False # By default we assume prior is uninformed (zero mean)
		self.posterior_dict = {}
		self.prediction_dict = {}

		if len(self.X) != 0:
			# prior mean and covar set to 0 and diagonal by default
			self.prior_mean = np.zeros((self.input_dim, 1))
			self.prior_covar = np.identity(self.input_dim) * self.prior_base_var * self.obs_noise_std**2

		# self.set_prior(self.prior_mean)

	def set_poly_transform(self, degree):
		'''
		Apply a Polynomial transformation
		'''
		self.transform = True
		self.degree = degree
		self.poly = PolynomialFeatures(degree=degree)
		if len(self.X) > 0: 
			self.X_trans = self.poly.fit_transform(self.X.T).T
			self.input_dim = len(self.X_trans)
			# Resize prior
			self.set_prior(0, 1e3) # TO DO change this!
	
	# Call to active homogenous coordinates
	def homogenize(self):
		self.flag_homogenize = True
		if len(self.X) > 0:
			self.X = np.vstack((self.X, np.ones(self.num_examples)))
			self.input_dim = len(self.X)
		return


	# Define custom prior for weights
	def set_informed_prior(self, mean, var):
		if isinstance(mean, (int, float)):
			self.prior_mean = mean * np.ones((self.input_dim, 1))
		else:
			self.prior_mean = mean.reshape(-1, 1)
		if isinstance(var, (int, float)):
			self.prior_covar = np.identity(self.input_dim) * var
		else: 
			self.prior_covar = var
		self.has_informed_prior = True

	def add_training_data(self, train_inputs, train_output_dict):
		'''
		Assumes train_inputs given in N x d or d
		'''
		# TO DO: Fix this, need to transpose here since I'm not at training anymore
		if len(self.X) == 0: # if uninitialized
			self.X = np.array(train_inputs)
			if len(self.X.shape) <= 1:
				self.X = self.X.reshape((-1, 1)) # Assums a 1D vector represents a set of N 1D examples
			self.num_examples = self.X.shape[0]
			self.input_dim = self.X.shape[1]
			if self.transform:
				self.X_trans = self.poly.fit_transform(self.X)
				self.input_dim = self.X_trans.shape[1]
			# if self.flag_homogenize:
			# 	self.homogenize()

			# Need to set prior if this is first time data is being added.
			if not self.has_informed_prior:
				self.prior_mean = np.zeros((self.input_dim, 1))
				self.prior_covar = np.identity(self.input_dim) * self.prior_base_var

			for metric, data in train_output_dict.items():
				self.y_dict[metric] = np.array(data).reshape((-1, 1))
		else: # Already have some data

			train_inputs = np.array(train_inputs)
			if len(train_inputs.shape) <= 1:
				train_inputs = train_inputs.reshape((-1, 1))
			self.X = np.concatenate((self.X, train_inputs), 0)
			self.num_examples = self.X.shape[0]
			if self.transform:
				self.X_trans = np.concatenate((self.X_trans, self.poly.transform(train_inputs)), 0)
			for metric, data in train_output_dict.items():
				data = np.array(data).reshape((-1, 1)) # convert to np column vector
				self.y_dict[metric] = np.concatenate((self.y_dict[metric], data), 0)
		
		# self.set_prior(0, 1e3) # TO DO change this!

	# Computes poseterior parameters from training data and prior 
	def train(self):
		# Math needs X to be in d x N format
		if self.transform:
			X = self.X_trans.T
		else:
			X = self.X.T

		for metric, y in self.y_dict.items():
			y = np.array(y).reshape((-1, 1))
			A = (X @ X.T / self.obs_noise_std**2
				+ np.linalg.inv(self.prior_covar))
			posterior_covar = np.linalg.inv(A)
			# print("X ", self.X.shape)
			# print("y ", self.y.shape)
			# print("prior covar ", self.prior_covar.shape)
			# print("prior mean ", self.prior_mean.shape)
			posterior_mean = (posterior_covar
								@ (X @ y / self.obs_noise_std**2
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
			test_input: array in N x d format
		'''
		test_input = np.array(test_input)
		if len(test_input.shape) <= 1:
			test_input = test_input.reshape((-1, 1))
		if self.transform:
			test_input = self.poly.transform(test_input)
		test_input = test_input.T # Needs to be in d x N format for computation
		for metric, (posterior_mean, posterior_covar) in self.posterior_dict.items():
			pred_mean = test_input.T @ posterior_mean
			pred_covar = test_input.T @ posterior_covar @ test_input
			self.prediction_dict[metric] = (pred_mean, np.diagonal(pred_covar))
			if prediction_df is not None:
				prediction_df[metric] = pred_mean
				prediction_df[metric+"_var"] = np.diagonal(pred_covar)

		return self.prediction_dict
