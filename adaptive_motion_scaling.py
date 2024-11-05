# Example of Adaptive Motion Scaling algorithm
from performance_modeling.performance_models import (
    BayesRegressionPerformanceModel,
)
from delay_modeling.distribution_estimation import DistributionEstimation
import pandas as pd
from collections import deque
import numpy as np

# Initialize user performance model
user_name = "user_xiao"
data_folder = "dvrk/trial_data"
# Load the performance metric data for the user
user_name = "user_xiao"
data_folder = "dvrk/trial_data"
user_data_file = f"{data_folder}/{user_name}_metric_data.csv"
data = pd.read_csv(user_data_file)
# list of metrics are all
metric = "time_score"  # Choose metric to evaluate
# Prepare training data
X = data[["latency", "scale"]]
Y = data[metric]

# Generate user performance model
user_performance_model = BayesRegressionPerformanceModel(
    X, Y, set_poly_transform=2
)
user_performance_model.train()

# For estimating the delay distribution
delay_estimator = DistributionEstimation()


# Initialize adaptive motion scaling
buffer = deque()
while running:

    new_message = receive_message()
    current_delay = (
        get_current_time() - packet.timestamp
    )  # get current delay (make sure system time is synchronized)

    # Update current estimation of delay, get desired effective delay
    delay_estimator.update(current_delay)
    desired_effective_delay = delay_estimator.get_value_at_percentile(90)

    # Get optimal scale from user model
    set_scale = user_performance_model.get_optimal_scale(
        desired_effective_delay,
        scale_domain=np.arange(0.1, 1.0, 0.1),
        metric="time_score",
    )

    # Inject delay to achieve desired effective delay
    inject_delay = (
        desired_effective_delay - current_delay
    )  # calculate delay to inject
    buffer.append(message)  # Add message to buffer

    # Publish messages while injected delay is not reached
    while (get_current_time() - buffer[-1].timestamp) > inject_delay:
        publish(buffer.popleft())
