# This is used to evaluate the performance model for a particular user.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from performance_models import (
    BayesRegressionPerformanceModel,
)


# Generates and trains a user performance model
def generate_user_model(data_filepath):
    data = pd.read_csv(data_filepath, index_col=0)
    metric_list = [
        col for col in data.columns if col not in ["latency", "scale"]
    ]
    X = data[["latency", "scale"]]
    Y_dict = data[metric_list].to_dict("list")
    model = BayesRegressionPerformanceModel(X, Y_dict, set_poly_transform=2)
    model.train()
    return model


if __name__ == "__main__":

    # Load the performance metric data for the user
    user_data_file = "example_user_data.csv"
    model = generate_user_model(user_data_file)

    # Choose domain for scaling factors
    scale_domain = [1, 2, 3, 4]

    # Get the optimal scaling factor in regards to completion time
    # for some levels of delay
    for delay in [2, 5]:
        optimal_scale = model.get_optimal_scale(
            delay, scale_domain=scale_domain, metric="time_score"
        )
        print(f"Optimal scale for delay {delay} is {optimal_scale}")
