import json
import numpy as np
import argparse

def total_variation_distance(p, q):
    return 0.5 * np.sum(np.abs(np.array(p) - np.array(q)))

def compare_params(original_file, estimated_file):
    with open(original_file, 'r') as f:
        orig = json.load(f)

    with open(estimated_file, 'r') as f:
        estim = json.load(f)

    Theta_orig = np.array(orig["Theta"])       # shape (4, w)
    ThetaB_orig = np.array(orig["ThetaB"])     # shape (4,)
    Theta_est = np.array(estim["Theta"])       # shape (4, w)
    ThetaB_est = np.array(estim["ThetaB"])     # shape (4,)
    w = Theta_orig.shape[1]

    tvd_list = []
    for j in range(w):
        tvd = total_variation_distance(Theta_orig[:, j], Theta_est[:, j])
        tvd_list.append(tvd)

    tvd_background = total_variation_distance(ThetaB_orig, ThetaB_est)
    tvd_avg = (np.sum(tvd_list) + tvd_background) / (w + 1)

    print("TVD for background (ThetaB):", round(tvd_background, 4))
    for j, val in enumerate(tvd_list):
        print(f"TVD for position {j+1} (Theta[:,{j}]):", round(val, 4))
    print("\nAverage total variation distance:", round(tvd_avg, 4))

# -------- main --------

parser = argparse.ArgumentParser(description="Compare estimated vs true parameters")
parser.add_argument('--original', default='params_set1.json', help='Plik z oryginalnymi parametrami')
parser.add_argument('--estimated', default='estimated_params.json', help='Plik z estymowanymi parametrami')
args = parser.parse_args()

compare_params(args.original, args.estimated)
