import pickle
import numpy as np
import matplotlib.pyplot as plt

data_folder = "controller_data_files/fixedTestSet_maxUnc"
with open(f"{data_folder}/eval_data.pkl", "rb") as f:
    eval_data = pickle.load(f)

mse_scores = eval_data["mse_scores"]
optimal_scale_errors = eval_data["optimal_scale_errors"]

fig, axes = plt.subplots(2, 2)
fig.suptitle("Evaluation Metrics for Max Uncertainty Policy")

axes[0, 0].plot(mse_scores["throughput"])
axes[0, 0].set_title("MSE for Throughput")

axes[0, 1].plot(mse_scores["total_error"])
axes[0, 1].set_title("MSE for Total Error")

axes[1, 0].plot(optimal_scale_errors["throughput"])
axes[1, 0].set_title("Optimal Scale Error for Throughput")

axes[1, 1].plot(optimal_scale_errors["total_error"])
axes[1, 1].set_title("Optimal Scale Error for Total Error")

plt.tight_layout()
plt.savefig(f"{data_folder}/eval_metric_plot.png")
plt.show()