import json
import argparse
from em_algorithm import em_algorithm
import numpy as np


def ParseArguments():
    parser = argparse.ArgumentParser(description="Motif estimator (EM)")
    parser.add_argument('--input', default="generated_data.json", required=False,
                        help='File with input data (default: %(default)s)')
    parser.add_argument('--output', default="estimated_params.json", required=False,
                        help='File where the estimated parameters will be saved (default: %(default)s)')
    parser.add_argument('--estimate-alpha', default="no", required=False,
                        help='Should alpha be estimated or not? (default: %(default)s)')
    args = parser.parse_args()
    return args.input, args.output, args.estimate_alpha

input_file, output_file, estimate_alpha_flag = ParseArguments()

with open(input_file, 'r') as inputfile:
    data = json.load(inputfile)

alpha = data['alpha']
data = np.asarray(data['data'])
k, w = data.shape

estimate_alpha = estimate_alpha_flag.lower() == "yes"

# EM estymacja
Theta, ThetaB, estimated_alpha = em_algorithm(data, alpha, w, estimate_alpha=estimate_alpha)

estimated_params = {
    "alpha": estimated_alpha if estimate_alpha else alpha,
    "Theta": Theta.tolist(),
    "ThetaB": ThetaB.tolist()
}

with open(output_file, 'w') as outfile:
    json.dump(estimated_params, outfile)