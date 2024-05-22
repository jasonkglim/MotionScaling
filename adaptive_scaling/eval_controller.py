import pickle
import numpy as np
import matplotlib.pyplot as plt

policy_types = ["maxUnc", "greedy", "random", "50greedy50maxUnc"] #, "GreedyRandomMaxUnc"]
policy_types_official = ["Min Entropy", "Greedy", "Random", "Balanced"]
fig_all, axes_all = plt.subplots(2, 2, figsize=(12, 6))
fig_all.suptitle(f"Policy Evaluation Metrics")

num_trials = 100

for policy_type, policy_name in zip(policy_types, policy_types_official):
    avg_mse_tp = []
    avg_mse_err = []
    avg_ose_tp = []
    avg_ose_err = []
    all_metric_avgs = []
    all_metric_optimums = []
    for i in range(num_trials):
        data_folder = f"controller_data_files/neweval/{policy_type}"

        with open(f"{data_folder}/trial{i}_eval_data.pkl", "rb") as f:
            eval_data = pickle.load(f)

        mse_scores = eval_data["mse_scores"]
        optimal_scale_errors = eval_data["optimal_scale_errors"]
        obs_metric_avgs = eval_data["obs_metric_avgs"]
        obs_metric_optimums = eval_data["obs_metric_optimums"]

        # fig, axes = plt.subplots(2, 2)
        # fig.suptitle(f"Evaluation Metrics for {policy_type} Policy")

        # axes[0, 0].plot(mse_scores["throughput"])
        # axes[0, 0].set_title("MSE for Throughput")

        # axes[0, 1].plot(mse_scores["total_error"])
        # axes[0, 1].set_title("MSE for Total Error")

        # axes[1, 0].plot(optimal_scale_errors["throughput"])
        # axes[1, 0].set_title("Optimal Scale Error for Throughput")

        # axes[1, 1].plot(optimal_scale_errors["total_error"])
        # axes[1, 1].set_title("Optimal Scale Error for Total Error")

        # fig.tight_layout()
        # fig.savefig(f"{data_folder}/eval_metric_plot.png")
        # plt.close()

        avg_mse_tp.append(mse_scores["throughput"])
        # avg_mse_err.append(mse_scores["total_error"])
        avg_ose_tp.append(optimal_scale_errors["throughput"])
        # avg_ose_err.append(optimal_scale_errors["total_error"])
        all_metric_avgs.append(obs_metric_avgs["throughput"])
        all_metric_optimums.append(obs_metric_optimums["throughput"])

    avg_mse_tp = np.mean(np.array(avg_mse_tp), axis=0)
    # avg_mse_err = np.mean(np.array(avg_mse_err), axis=0)
    avg_ose_tp = np.mean(np.array(avg_ose_tp), axis=0)
    # avg_ose_err = np.mean(np.array(avg_ose_err), axis=0)
    avg_metric_avgs = np.mean(np.array(all_metric_avgs), axis=0)
    avg_metric_optimums = np.mean(np.array(all_metric_optimums), axis=0)

    axes_all[0, 0].plot(avg_mse_tp, label=policy_name, marker='o')
    axes_all[0, 0].set_title("MSE for Throughput")
    axes_all[0, 0].set_xlabel("Training Points")

    # axes_all[0, 1].plot(avg_mse_err, label=policy_type)
    # axes_all[0, 1].set_title("MSE for Total Error")

    axes_all[0, 1].plot(avg_ose_tp, label=policy_name, marker='o')
    axes_all[0, 1].set_title("Optimal Scale Error for Throughput")
    axes_all[0, 1].set_xlabel("Training Points")

    axes_all[1, 0].plot(avg_metric_avgs, label=policy_name, marker='o')
    axes_all[1, 0].set_title("Average Observed Throughput")
    axes_all[1, 0].set_xlabel("Training Points")

    axes_all[1, 1].plot(avg_metric_optimums, label=policy_name, marker='o')
    axes_all[1, 1].set_title("Maximum Observed Throughput")
    axes_all[1, 1].set_xlabel("Training Points")


axes_all[0, 0].legend()
fig_all.tight_layout()
fig_all.savefig(f"controller_data_files/all_policies_neweval.png")
plt.show()