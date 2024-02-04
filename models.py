### Contains functions for different modeling approaches
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, RationalQuadratic
import numpy as np

# Object for model results


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
class BayesRegression:
	def __init__(self, train_input, train_output, noise=0):
		self.X = np.array(train_input)
		self.y = np.array(train_output).reshape((-1, 1))
		self.input_dim = self.X.shape[0]
		self.num_examples = self.X.shape[1]
		self.prior_mean = np.zeros((self.input_dim, 1))
		self.prior_covar = np.identity(self.input_dim)
		self.noise = noise # Defines the variance of gaussian observation noise

	# Define custom prior for weights
	def set_prior(self, mean, var):
		self.prior_mean = mean.reshape(-1, 1)
		if isinstance(var, int):
			self.prior_covar = np.identity(self.input_dim) * var
		else: 
			self.prior_covar = var

	# Computes poseterior parameters from training data and prior 
	def fit(self):
		A = (self.X @ self.X.T / self.noise**2
	   		 + np.linalg.inv(self.prior_covar))
		self.posterior_covar = np.linalg.inv(A)
		# print(self.posterior_covar.shape)
		# print("X ", self.X.shape)
		# print("y ", self.y.shape)
		# print("prior covar ", self.prior_covar.shape)
		# print("prior mean ", self.prior_mean.shape)
		self.posterior_mean = (self.posterior_covar
						 	   @ (self.X @ self.y / self.noise**2
			  					  + np.linalg.inv(self.prior_covar)
								  @ self.prior_mean))

		# print("post mean ", self.posterior_mean.shape)
		# print("post covar ", self.posterior_covar.shape)
		return self.posterior_mean, self.posterior_covar
	
	# Compute mean and covariance of predictive distribution
	def predict(self, test_input):
		test_input = np.array(test_input)
		self.pred_mean = test_input.T @ self.posterior_mean
		self.pred_covar = test_input.T @ self.posterior_covar @ test_input
		return self.pred_mean, self.pred_covar
