import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from utils import stratified_sample, annotate, even_train_split
import glob  # Importing the glob module to find all the files matching a pattern
from models import BayesRegression
import scaling_policy

# Pattern to match the data files
file_pattern = "data_files/user_*/metric_df.csv"

# Initialize a dictionary to store one_user_one_user_dataframes for each dataset
all_datasets = {}

# Loop through each file that matches the file pattern
for filepath in glob.glob(file_pattern):
    # print(filepath)
    # print(filepath.split('/'))
    user_name = filepath.split('/')[1]
    # user_name = filepath.split('\\')[1]
    print(f"Processing {filepath} dataset...")

    # Read in data file as a pandas dataframe
    data = pd.read_csv(filepath, index_col=0)

    # add weighted performance metric
    w = 1
    data["total_error"] = data['avg_osd'] + data['avg_target_error']
    data["weighted_performance"] = 10*data['throughput'] - w*data["total_error"]

    all_datasets[user_name] = data

# Combine datasets for Lizzie
lizzie1 = all_datasets["user_lizzie1"]
lizzie2 = all_datasets["user_lizzie2"]
combined_df = pd.concat([lizzie1, lizzie2])
all_datasets["user_lizzie"] = combined_df.groupby(['latency', 'scale']).mean().reset_index()

## Start modeling
player = "user_jason"
player_data = all_datasets[player]
metric_dict = {}

# Control loop:
for i in range(len(player_data)):
    # obs_data = player_data[control_scale, latency]

    # Evaluate obs data?

    # add obs_data to training data of PerformanceModel
    model = BayesRegression()
    model.add_training_data(obs_data)
    model.train()

    # Input Performance Model to Scaling Policy
    ScalingPolicy.update(PerformanceModel)
    control_scale = ScalingPolicy.get_scale()




    


