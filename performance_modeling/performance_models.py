from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, RationalQuadratic
import numpy as np
import matplotlib.pyplot as plt


class PerformanceModel:
    def __init__(self, train_inputs=None, train_output_dict=None):
        '''
        Base class for modeling performance metrics.

        Args:
            train_inputs (array-like, optional): Input data of shape (N, d).
            train_output_dict (dict, optional): Dictionary with output values for each metric.
        '''
        self.X = self._initialize_inputs(train_inputs)
        self.num_examples = self.X.shape[0] if self.X.size else 0
        self.input_dim = self.X.shape[1] if self.X.size else 0
        self.y_dict = self._initialize_outputs(train_output_dict)

    def _initialize_inputs(self, train_inputs):
        '''Initialize and validate input data.'''
        if train_inputs is None or len(train_inputs) == 0:
            return np.array([])
        train_inputs = np.array(train_inputs)
        if train_inputs.ndim == 1:
            train_inputs = train_inputs.reshape(-1, 1)
        return train_inputs

    def _initialize_outputs(self, train_output_dict):
        '''Initialize and validate output dictionary.'''
        y_dict = {}
        if train_output_dict is None:
            return y_dict
        for metric, data in train_output_dict.items():
            data = np.array(data)
            if data.shape[0] != self.num_examples:
                raise ValueError(f"Mismatch between input examples and output examples for metric '{metric}'")
            y_dict[metric] = data.reshape(-1, 1)
        return y_dict

    def add_metric(self, metric_name, output_data):
        '''
        Add a new metric and its corresponding output data.

        Args:
            metric_name (str): Name of the metric.
            output_data (array-like): Output data to add.
        '''
        output_data = np.array(output_data)
        if output_data.shape[0] != self.num_examples:
            raise ValueError("Mismatch between input examples and output examples for the new metric.")
        self.y_dict[metric_name] = output_data.reshape(-1, 1)


class BayesRegressionPerformanceModel(PerformanceModel):
    '''
    Bayesian Regression with a Normal-Inverse Gamma prior.

    Args:
        hyperparams (tuple, optional): Contains (m, V, d, a) for Normal-Inverse Gamma distribution.
            - m and V: mean and covariance of weight vector
            - d and a: shape and rate parameters for noise variance
    '''
    def __init__(self, train_inputs=None, train_output_dict=None, hyperparams=None):
        super().__init__(train_inputs, train_output_dict)
        
        # Set hyperparameters
        if hyperparams is None:
            self.hyperparams = self._default_hyperparams()
            self.inform_prior = False  # Use reference prior if no hyperparams provided
        else:
            self.hyperparams = hyperparams
            self.inform_prior = True
        
        self.posterior_dict = {}
        self.prediction_dict = {}

    def _default_hyperparams(self):
        '''Return default hyperparameters for non-informative prior.'''
        dim = self.X.shape[1]
        m_0 = np.zeros((dim, 1))
        V_0 = np.inf
        d_0 = -dim
        a_0 = 0
        return [m_0, V_0, d_0, a_0]

    def train(self):
        '''
        Train the Bayesian regression model by updating posterior parameters for each metric.
        Returns:
            dict: Posterior parameters for each metric.
        '''
        m, V, d, a = self._unpack_hyperparams(self.hyperparams)
        X = self.X
        n, dim = X.shape

        V_inv = np.linalg.inv(V) if self.inform_prior else np.zeros((dim, dim))
        if not self.inform_prior and n < dim:
            V_inv += 0.001 * np.eye(dim)  # Handle singularity if n < d

        V_post = np.linalg.inv(V_inv + X.T @ X)
        d_post = d + n

        for metric, y in self.y_dict.items():
            self.posterior_dict[metric] = self._update_posterior(m, V_post, V_inv, X, y, a, d_post)

        return self.posterior_dict

    def _update_posterior(self, m, V_post, V_inv, X, y, a, d_post):
        '''Calculate posterior parameters for a single metric.'''
        m_post = V_post @ (V_inv @ m + X.T @ y) if self.inform_prior else V_post @ X.T @ y
        if self.inform_prior:
            a_post = a + m.T @ V_inv @ m + y.T @ y - m_post.T @ np.linalg.inv(V_post) @ m_post
        else:
            a_post = y.T @ y - m_post.T @ (X.T @ X) @ m_post

        return [m_post, V_post, d_post, a_post]

    def predict(self, test_input, prediction_df=None):
        '''
        Predicts output and variance for each metric given new input.

        Args:
            test_input (ndarray): New input data.
            prediction_df (pd.DataFrame, optional): DataFrame to store predictions.
        
        Returns:
            dict: Prediction parameters for each metric.
        '''
        for metric, params in self.posterior_dict.items():
            pred_loc, pred_var = self._compute_prediction(test_input, params)
            self.prediction_dict[metric] = (params[2], pred_loc, pred_var)

            if prediction_df is not None:
                prediction_df[metric] = pred_loc
                prediction_df[metric + "_var"] = pred_var

        return self.prediction_dict

    def _compute_prediction(self, test_input, params):
        '''Compute prediction mean and variance for a single metric.'''
        pred_loc = test_input @ params[0]
        pred_covar = params[3] * (np.eye(test_input.shape[0]) + test_input @ params[1] @ test_input.T)
        return pred_loc, np.diagonal(pred_covar)

    def _unpack_hyperparams(self, hyperparams):
        '''Unpack hyperparameters for readability.'''
        return hyperparams[0].reshape(-1, 1), hyperparams[1], hyperparams[2], hyperparams[3]
