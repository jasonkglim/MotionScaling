# Example of Adaptive Motion Scaling algorithm
from performance_modeling.performance_models import BayesRegressionPerformanceModel
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

user_performance_model = BayesRegressionPerformanceModel(X, Y)
delay_estimator = OnlineHistogram()

buffer = deque()

# Initialize adaptive motion scaling
while True:
    packet = receive_packet() # receive new packet
    current_delay = get_current_time() - packet.timestamp # get current delay
    delay_estimator.update(current_delay) # Update current estimation of latency distribution
    desired_effective_delay = delay_estimator.get_value_at_percentile(90) # get desired effective delay
    set_scale = user_performance_model.get_optimal_scale(desired_effective_delay) # get optimal scale from user model
    inject_delay = desired_effective_delay - current_delay # calculate delay to inject
    buffer.append(packet) # Add packet to command buffer
    while( (get_current_time() - buffer[-1].timestamp) < inject_delay):
        publish(buffer.popleft())