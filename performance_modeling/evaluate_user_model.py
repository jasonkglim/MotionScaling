# This is used to evaluate the performance model for a particular user.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from performance_models import (
    BayesRegressionPerformanceModel,
)

if __name__ == "__main__":

    # Load the performance metric data for the user
    user_name = "user_xiao"
    data_folder = "../dvrk/trial_data"
    user_data_file = f"{data_folder}/{user_name}_metric_data.csv"
    data = pd.read_csv(user_data_file)
    # Print column headers that are not "latency" or "scale"
    metric_list = [
        col for col in data.columns if col not in ["latency", "scale"]
    ]
    print(f"Metrics available: {metric_list}")

    # Prepare data
    X = data[["latency", "scale"]]
    Y = data[metric_list]
    Y_dict = data[metric_list].to_dict("list")
    model = BayesRegressionPerformanceModel(X, Y_dict)
    model.train()

    # Choose domain for scaling factors
    scale_domain = np.arange(0.1, 1.6, 0.1)

    # Get the optimal scaling factor in regards to completion time
    # for some levels of delay
    for delay in [0.1, 0.5, 1.0, 2.0]:
        optimal_scale = model.get_optimal_scale(
            delay, scale_domain=scale_domain, metric="time_score"
        )
        print(f"Optimal scale for delay {delay} is {optimal_scale}")
