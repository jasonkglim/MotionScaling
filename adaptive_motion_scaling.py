# Example of Adaptive Motion Scaling algorithm
from performance_modeling.performance_models import BayesRegressionPerformanceModel, generate_user_model
from delay_modeling.distribution_estimation import OnlineHistogram
import pandas as pd
import deque

# Initialize user performance model
user_name = "user_xiao"
data_folder = "dvrk/trial_data"
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

# Generate user performance model
user_performance_model = BayesRegressionPerformanceModel(X, Y)

# For estimating the delay distribution
delay_estimator = OnlineHistogram()


# Initialize adaptive motion scaling
buffer = deque()
while running:

    new_message = receive_message() # receive new packet
    current_delay = get_current_time() - packet.timestamp # get current delay

    # Update current estimation of delay, get desired effective delay
    delay_estimator.update(current_delay) # Update current estimation of latency distribution
    desired_effective_delay = delay_estimator.get_value_at_percentile(90) # get desired effective delay

    # Get optimal scale from user model
    set_scale = user_performance_model.get_optimal_scale(desired_effective_delay)

    # Inject delay to achieve desired effective delay
    inject_delay = desired_effective_delay - current_delay # calculate delay to inject
    buffer.append(message) # Add message to buffer

    # Publish messages while injected delay is not reached
    while( (get_current_time() - buffer[-1].timestamp) > inject_delay):
        publish(buffer.popleft())