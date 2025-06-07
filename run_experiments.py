import json
import numpy as np
import argparse
import os
from em_algorithm import em_algorithm
import matplotlib.pyplot as plt

def total_variation_distance(p, q):
    return 0.5 * np.sum(np.abs(np.array(p) - np.array(q)))

def generate_random_params(w):
    Theta = np.random.dirichlet([1]*4, size=w).T
    ThetaB = np.random.dirichlet([1]*4)
    return Theta, ThetaB

def generate_data(k, w, alpha, Theta, ThetaB):
    X = []
    for _ in range(k):
        if np.random.rand() < alpha:
            seq = [np.random.choice([1, 2, 3, 4], p=Theta[:, j]) for j in range(w)]
        else:
            seq = [np.random.choice([1, 2, 3, 4], p=ThetaB) for _ in range(w)]
        X.append(seq)
    return np.array(X)

def compute_average_tvd(Theta_true, ThetaB_true, Theta_est, ThetaB_est):
    w = Theta_true.shape[1]
    tvd_list = [total_variation_distance(Theta_true[:,j], Theta_est[:,j]) for j in range(w)]
    tvd_b = total_variation_distance(ThetaB_true, ThetaB_est)
    return (np.sum(tvd_list) + tvd_b) / (w + 1)

def run_multiple_experiments(w, k, alpha, n, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tvd_scores = []

    for i in range(n):
        Theta_true, ThetaB_true = generate_random_params(w)
        X = generate_data(k, w, alpha, Theta_true, ThetaB_true)
        Theta_est, ThetaB_est, _ = em_algorithm(X, alpha, w, estimate_alpha=False)

        tvd = compute_average_tvd(Theta_true, ThetaB_true, Theta_est, ThetaB_est)
        tvd_scores.append(tvd)

        # zapis (opcjonalny)
        with open(os.path.join(output_dir, f"run_{i+1}.json"), "w") as f:
            json.dump({
                "tvd": tvd,
                "Theta": Theta_est.tolist(),
                "ThetaB": ThetaB_est.tolist()
            }, f)

    print(f"\nResults after {n} repetitions:")
    print(f"Average TVD: {np.mean(tvd_scores):.4f}")
    print(f"Standard deviation: {np.std(tvd_scores):.4f}")

    plt.hist(tvd_scores, bins=10, color='skyblue', edgecolor='black')
    plt.title(f"Histogram TVD (w={w}, k={k}, alpha={alpha})")
    plt.xlabel("TVD")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "histogram_tvd.png"))
    plt.close()

# ----------- main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple EM experiments for fixed w, k")
    parser.add_argument('--w', type=int, default=5, help='Length of motif (w)')
    parser.add_argument('--k', type=int, default=50, help='Number of sequences (k)')
    parser.add_argument('--alpha', type=float, default=0.5, help='Proportion of motif sequences')
    parser.add_argument('--n', type=int, default=10, help='Number of repetitions')
    parser.add_argument('--output-dir', default="experiment_results", help='Where to save per-run results')
    args = parser.parse_args()

    run_multiple_experiments(args.w, args.k, args.alpha, args.n, args.output_dir)