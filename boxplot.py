import numpy as np
import matplotlib.pyplot as plt
from em_algorithm import em_algorithm
from run_experiments import generate_data, generate_random_params, compute_average_tvd
from collections import defaultdict


def experiment_grid(k_values, w_values, alpha_values, n_repeats=50):
    results = []
    for alpha in alpha_values:
        for w in w_values:
            for k in k_values:
                tvds = []
                for _ in range(n_repeats):
                    Theta_true, ThetaB_true = generate_random_params(w)
                    X = generate_data(k, w, alpha, Theta_true, ThetaB_true)
                    Theta_est, ThetaB_est, _ = em_algorithm(X, alpha, w, estimate_alpha=False)
                    tvd = compute_average_tvd(Theta_true, ThetaB_true, Theta_est, ThetaB_est)
                    tvds.append(tvd)
                avg = np.mean(tvds)
                std = np.std(tvds)
                results.append({
                    "k": k, "w": w, "alpha": alpha,
                    "mean_tvd": avg, "std_tvd": std,
                    "tvds": tvds
                })
    return results


def plot_boxplot_tvd_vs_w(results, k_fixed=200, alpha_fixed=0.5):
    grouped = defaultdict(list)
    for r in results:
        if r['k'] == k_fixed and r['alpha'] == alpha_fixed:
            grouped[r['w']] = r['tvds']

    w_vals = sorted(grouped.keys())
    data = [grouped[w] for w in w_vals]

    plt.figure()
    box = plt.boxplot(data, labels=w_vals, patch_artist=True)
    for patch in box['boxes']:
        patch.set_facecolor('lightblue')
    plt.title(f"Boxplot of TVD vs w (k={k_fixed}, alpha={alpha_fixed})")
    plt.xlabel("Motif length (w)")
    plt.ylabel("TVD")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"boxplot_tvd_vs_w_k{k_fixed}.png")
    plt.close()


def plot_boxplot_tvd_vs_k(results, w_fixed=5, alpha_fixed=0.5):
    grouped = defaultdict(list)
    for r in results:
        if r['w'] == w_fixed and r['alpha'] == alpha_fixed:
            grouped[r['k']] = r['tvds']

    k_vals = sorted(grouped.keys())
    data = [grouped[k] for k in k_vals]

    plt.figure()
    box = plt.boxplot(data, labels=k_vals, patch_artist=True)
    for patch in box['boxes']:
        patch.set_facecolor('lightblue')
    plt.title(f"Boxplot of TVD vs k (w={w_fixed}, alpha={alpha_fixed})")
    plt.xlabel("Number of sequences (k)")
    plt.ylabel("TVD")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"boxplot_tvd_vs_k_w{w_fixed}.png")
    plt.close()


def plot_tvd_vs_k(results, alpha_fixed=0.5):
    grouped = defaultdict(list)
    for r in results:
        if r['alpha'] == alpha_fixed:
            grouped[r['w']].append(r)

    plt.figure()
    for w, entries in grouped.items():
        entries.sort(key=lambda x: x['k'])
        k_vals = [r['k'] for r in entries]
        mean_tvds = [r['mean_tvd'] for r in entries]
        std_tvds = [r['std_tvd'] for r in entries]
        plt.errorbar(k_vals, mean_tvds, yerr=std_tvds, fmt='-o', capsize=5, label=f"w={w}")

    plt.title(f"Average TVD vs k (alpha={alpha_fixed})")
    plt.xlabel("Number of sequences (k)")
    plt.ylabel("Average TVD")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"tvd_vs_k_alpha{int(alpha_fixed*100)}.png")
    plt.close()


def plot_tvd_vs_w(results, alpha_fixed=0.5):
    grouped = defaultdict(list)
    for r in results:
        if r['alpha'] == alpha_fixed:
            grouped[r['k']].append(r)

    plt.figure()
    for k, entries in grouped.items():
        entries.sort(key=lambda x: x['w'])
        w_vals = [r['w'] for r in entries]
        mean_tvds = [r['mean_tvd'] for r in entries]
        std_tvds = [r['std_tvd'] for r in entries]
        plt.errorbar(w_vals, mean_tvds, yerr=std_tvds, fmt='-o', capsize=5, label=f"k={k}")

    plt.title(f"Average TVD vs w (alpha={alpha_fixed})")
    plt.xlabel("Motif length (w)")
    plt.ylabel("Average TVD")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"tvd_vs_w_alpha{int(alpha_fixed*100)}.png")
    plt.close()


def plot_tvd_vs_alpha(results):
    from random import sample
    grouped = defaultdict(list)
    for r in results:
        grouped[(r['k'], r['w'])].append(r)

    selected_combinations = sample(list(grouped.keys()), max(1, len(grouped)//4))

    plt.figure()
    for (k, w) in selected_combinations:
        entries = grouped[(k, w)]
        entries.sort(key=lambda x: x['alpha'])
        a_vals = [r['alpha'] for r in entries]
        mean_tvds = [r['mean_tvd'] for r in entries]
        std_tvds = [r['std_tvd'] for r in entries]
        plt.errorbar(a_vals, mean_tvds, yerr=std_tvds, fmt='-o', capsize=5, label=f"k={k}, w={w}")

    plt.title(f"Average TVD vs alpha (subset of configurations)")
    plt.xlabel("Proportion alpha")
    plt.ylabel("Average TVD")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"tvd_vs_alpha_subset.png")
    plt.close()


if __name__ == "__main__":
    k_vals = [10, 20, 50, 100, 200]
    w_vals = [3, 5, 7, 10]
    alpha_vals = [0.1, 0.3, 0.5, 0.7, 0.9]

    print("Running experiments (this may take a while)...")
    results = experiment_grid(k_vals, w_vals, alpha_vals, n_repeats=50)


    print("Plotting boxplots...")
    plot_boxplot_tvd_vs_w(results, k_fixed=20, alpha_fixed=0.5)
    plot_boxplot_tvd_vs_k(results, w_fixed=5, alpha_fixed=0.5)
    plot_boxplot_tvd_vs_w(results, k_fixed=200, alpha_fixed=0.5)
    plot_boxplot_tvd_vs_k(results, w_fixed=10, alpha_fixed=0.5)

    print("All plots saved.")
