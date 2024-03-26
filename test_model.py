from models import BayesRegression
import numpy as np
import matplotlib.pyplot as plt
import os

# Create input
X = np.array([(np.floor(i/5)+1)*0.1 for i in range(40)])
np.random.seed()
# noise = np.array([np.random.normal(scale=s) for s in np.random.randint(1, 8, size=len(X))])
noise = np.random.normal(scale=0.2, size=len(X))
f = X*0.25 + 3
y = f + noise
# for x in X:
# 	scale = np.random.randint(1, 5)
# 	noise = np.random.normal(loc=0, scale=scale)
# 	y.append(x * 2 + 1 + noise)

X_homo = np.vstack((X, np.ones(X.shape)))
XY = np.vstack((y, X))
np.random.shuffle(XY.T)
y_train = XY[0,:]
X_train = XY[1:,:]
model = BayesRegression()
model.set_poly_transform(degree=2)
print(noise)
os.makedirs("figures/BayesTestPoly_idk", exist_ok=True)
# test_input = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
test_input = np.linspace(0.1, 0.8, 100)
for i in range(len(X_train[0])):
	model.add_training_data(X_train[:,i].reshape(-1, 1), {'y': y_train[i].reshape(-1, 1)})
	model.train()
	prediction_dict = model.predict(test_input=test_input)
	ypred_mean = prediction_dict['y'][0].flatten()
	ypred_var = prediction_dict['y'][1].flatten()
	plt.fill_between(test_input, ypred_mean-ypred_var, ypred_mean+ypred_var, alpha = 0.3)
	plt.fill_between(test_input, ypred_mean-2*ypred_var, ypred_mean+2*ypred_var, alpha = 0.1)
	plt.scatter(X_train.flatten()[:i+1], y_train[:i+1], label='Observations')
	plt.plot(X, f, color='black', linestyle='--', label="True Function")
	plt.plot(test_input, ypred_mean, color='red', linestyle='-.', label="Predicted Mean")
	plt.grid(True)
	plt.savefig(f'figures/BayesTestPoly_idk/{i}.png')
	plt.close()