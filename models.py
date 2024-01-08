### Contains functions for different modeling approaches
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Object for model results


## Polynomail Regression
def PolyRegression(X, X_dense, X_train, Y_train, degree = 2):

	# Feature transformation
	poly = PolynomialFeatures(degree=degree)
	X_poly = poly.fit_transform(X)
	
	# Train model on training set
	model = LinearRegression()
	model.fit(poly.transform(X_train), Y_train)

	# Predict over original inputs
	Y_pred = model.predict(X_poly)

	# Predict over dense inputs
	Y_pred_dense = model.predict(poly.transform(X_dense))

	# Return predictions
	return {'Y_pred': Y_pred, 'Y_pred_dense': Y_pred_dense}





