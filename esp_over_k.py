import numpy as np
import matplotlib.pyplot as plt
from em_algorithm import em_algorithm
from run_experiments import generate_data, generate_random_params, compute_average_tvd

def experiment_over_k_values(w, alpha, k_values, n_per_k):
    avg_tvds = []
    std_tvds = []

    for k in k_values:
        tvds = []
        for _ in range(n_per_k):
            Theta_true, ThetaB_true = generate_random_params(w)
            X = generate_data(k, w, alpha, Theta_true, ThetaB_true)
            Theta_est, ThetaB_est, _ = em_algorithm(X, alpha, w, estimate_alpha=False)
            tvd = compute_average_tvd(Theta_true, ThetaB_true, Theta_est, ThetaB_est)
            tvds.append(tvd)
        avg_tvds.append(np.mean(tvds))
        std_tvds.append(np.std(tvds))

    return avg_tvds, std_tvds

# ----------- MAIN -----------
# if __name__ == "__main__":
#     w = 5
#     alpha = 0.8
#     k_values = [10, 20, 50, 100, 200, 500]
#     n_per_k = 10
#
#     avg_tvds, std_tvds = experiment_over_k_values(w, alpha, k_values, n_per_k)
#
#     # Wykres
#     plt.errorbar(k_values, avg_tvds, yerr=std_tvds, fmt='-o', capsize=5, color='darkblue')
#     plt.title(f"Average TVD vs k (w = {w}, alpha = {alpha})")
#     plt.xlabel("Number of sequences (k)")
#     plt.ylabel("Average TVD")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig("tvd_vs_k.png")
#     plt.show()