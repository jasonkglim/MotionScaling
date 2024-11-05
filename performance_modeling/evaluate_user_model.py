# This is used to evaluate the performance model for a particular user.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from performance_modeling.performance_models import BayesRegressionPerformanceModel

if __name__ == '__main__':

    # Load the performance metric data for the user
    user_name = "user_xiao"
    data_folder = "dvrk/trial_data"
    user_data_file = f"{data_folder}/{user_name}_metric_data.csv"
    data = pd.read_csv(user_data_file)
    # list of metrics are all 
    metric = "time_score" # Choose metric to evaluate

    # Prepare data 
    X = data[['latency', 'scale']]
    Y = data[metric]
    Y_dict = data[[metric]].to_dict('list')
    # poly = PolynomialFeatures(degree=1)
    # X_poly = poly.fit_transform(X.values)
    # print("X_Poly shape: ", X_poly.shape)
    # save posterior information from full dataset for later use
    model_full = BayesRegressionPerformanceModel(X, Y_dict)
    posterior_hyperparams = model_full.train()[metric]

    # Initialize evaluation metrics
    optimal_match_rate = []
    optimal_scale_error = []
    mse_scores = []
    full_mse_scores = []
    n_train_mse = []
    n_train_full_mse = []
    n_train_p = []
    
    # print('hi')
    n = len(data)
    n_train_values = range(2, n-1)
    for n_train in n_train_values:

        n_train_p.append(n_train / n)
        # Split into training/test sets
        # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=n_train/n)
        train_set, test_set = even_train_split(data, n_train)
        X_train, X_test = train_set[['latency', 'scale']], test_set[['latency', 'scale']]
        Y_train, Y_test = train_set[metric], test_set[metric]
        Y_train_dict = train_set[[metric]].to_dict('list')
        Y_test_dict = test_set[[metric]].to_dict('list')
        
        # Create dense test input
        # latency_set = data['latency'].unique()# np.arange(0.0, 0.76, 0.01)
        # latency_range = np.array(data['latency'].unique()) #np.linspace(latency_set.min(), latency_set.max(), 50)
        latency_range = np.arange(0.0, data['latency'].max()+0.01, 0.01)
        scale_range = np.arange(data['scale'].min(), data['scale'].max()+0.025, 0.025) #np.linspace(data['scale'].min(), data['scale'].max(), 50)
        latency_grid, scale_grid = np.meshgrid(latency_range, scale_range)
        X_dense = np.c_[latency_grid.ravel(), scale_grid.ravel()]
        X_dense = np.round(X_dense, 3)
            
        
        # X_train_poly = poly.transform(X_train.values)
        # X_test_poly = poly.transform(X_test.values)
        # X_dense_poly = poly.transform(X_dense)
        
        # Train model
        model = BayesRegressionPerformanceModel(X_train, Y_train_dict, set_poly_transform=2)
        posterior_dict = model.train()
        post_mean, post_covar = posterior_dict[metric][0], posterior_dict[metric][1]
        model_params = f"coef: {post_mean.flatten()}"
        # Predict
        Y_pred = model.predict(X)[metric][1]
        Y_pred_dense = model.predict(X_dense)[metric][1]

        ## Evaluate metrics
        dense_df = pd.DataFrame({
                'latency': X_dense[:, 0].flatten(),
                'scale': X_dense[:, 1].flatten(),
                'Y_pred_dense': Y_pred_dense.flatten()
            })
        data["Y_pred"] = Y_pred

        # Mean Square Error on whole dataset
        full_mse = mean_squared_error(Y, Y_pred)
        if True: #full_mse < 5000:
            n_train_full_mse.append(n_train)
            full_mse_scores.append(full_mse)

        # Mean Square Error on test set
        Y_test_pred = data.loc[Y_test.index]["Y_pred"]
        mse = mean_squared_error(Y_test, Y_test_pred)
        if True: #mse < 5000:
            n_train_mse.append(n_train)
            mse_scores.append(mse)
        
        if metric in ["throughput_standard", "avg_movement_speed_standard", "weighted_performance_standard"]: # optimal scale at maximum
            optimal_scale_dense = dense_df.loc[dense_df.groupby('latency')['Y_pred_dense'].idxmax()][['latency', 'scale']]
            optimal_scale_ref = data.loc[data.groupby('latency')[metric].idxmax()][['latency', 'scale']]
            optimal_scale_pred = data.loc[data.groupby('latency')['Y_pred'].idxmax()][['latency', 'scale']]
        else: # optimal scale at minimum
            optimal_scale_dense = dense_df.loc[dense_df.groupby('latency')['Y_pred_dense'].idxmin()][['latency', 'scale']]
            optimal_scale_ref = data.loc[data.groupby('latency')[metric].idxmin()][['latency', 'scale']]
            optimal_scale_pred = data.loc[data.groupby('latency')['Y_pred'].idxmin()][['latency', 'scale']]

        # Merge the results on 'latency'
        merged_ref_pred = pd.merge(optimal_scale_ref, optimal_scale_pred, 
                            on='latency', suffixes=('_ref', '_pred'))
        
        merged_ref_dense = pd.merge(optimal_scale_ref, optimal_scale_dense, 
                            on='latency', suffixes=('_ref', '_dense'))
        # print(optimal_scale_dense)
        # print(merged_ref_dense)
        

        # Count the number of matches
        matches = (merged_ref_pred['scale_ref'] == merged_ref_pred['scale_pred']).sum()
        scale_error = np.abs(merged_ref_dense['scale_ref'] - merged_ref_dense['scale_dense']).mean()

        optimal_match_rate.append(matches / len(optimal_scale_ref))
        optimal_scale_error.append(scale_error)

        # # Visualize model prediction
        # if n_train == n-2:
        #     model_heatmaps(data, dense_df, X_train, user, metric, model_type, model_params)

