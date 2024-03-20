import pickle
import numpy as np
import matplotlib.pyplot as plt

policy_types = ["maxUnc", "Greedy", "Random", "50greedy50maxUnc", "GreedyRandomMaxUnc"]

fig_all, axes_all = plt.subplots(2, 2)
fig_all.suptitle(f"Policy Evaluation Metrics")

for policy_type in policy_types:
    avg_mse_tp = []
    avg_mse_err = []
    avg_ose_tp = []
    avg_ose_err = []
    for i in range(5):
        data_folder = f"controller_data_files/fixedTestSet{i}_{policy_type}"

        with open(f"{data_folder}/eval_data.pkl", "rb") as f:
            eval_data = pickle.load(f)

        mse_scores = eval_data["mse_scores"]
        optimal_scale_errors = eval_data["optimal_scale_errors"]

        fig, axes = plt.subplots(2, 2)
        fig.suptitle(f"Evaluation Metrics for {policy_type} Policy")

        axes[0, 0].plot(mse_scores["throughput"])
        axes[0, 0].set_title("MSE for Throughput")

        axes[0, 1].plot(mse_scores["total_error"])
        axes[0, 1].set_title("MSE for Total Error")

        axes[1, 0].plot(optimal_scale_errors["throughput"])
        axes[1, 0].set_title("Optimal Scale Error for Throughput")

        axes[1, 1].plot(optimal_scale_errors["total_error"])
        axes[1, 1].set_title("Optimal Scale Error for Total Error")

        fig.tight_layout()
        fig.savefig(f"{data_folder}/eval_metric_plot.png")
        plt.close()

        avg_mse_tp.append(mse_scores["throughput"])
        avg_mse_err.append(mse_scores["total_error"])
        avg_ose_tp.append(optimal_scale_errors["throughput"])
        avg_ose_err.append(optimal_scale_errors["total_error"])

    avg_mse_tp = np.mean(np.array(avg_mse_tp), axis=0)
    avg_mse_err = np.mean(np.array(avg_mse_err), axis=0)
    avg_ose_tp = np.mean(np.array(avg_ose_tp), axis=0)
    avg_ose_err = np.mean(np.array(avg_ose_err), axis=0)

    axes_all[0, 0].plot(avg_mse_tp, label=policy_type)
    axes_all[0, 0].set_title("MSE for Throughput")

    axes_all[0, 1].plot(avg_mse_err, label=policy_type)
    axes_all[0, 1].set_title("MSE for Total Error")

    axes_all[1, 0].plot(avg_ose_tp, label=policy_type)
    axes_all[1, 0].set_title("Optimal Scale Error for Throughput")

    axes_all[1, 1].plot(avg_ose_err, label=policy_type)
    axes_all[1, 1].set_title("Optimal Scale Error for Total Error")

axes_all[0, 0].legend()
fig_all.tight_layout()
fig_all.savefig(f"controller_data_files/all_policies_avg_plot.png")
plt.show()