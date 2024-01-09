### Contains functions for different modeling approaches
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, RationalQuadratic

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

	# Return predictions
	return Y_pred


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

	return pred_mean, pred_std







