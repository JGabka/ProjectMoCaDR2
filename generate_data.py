import json
import argparse
import numpy as np


def generate_sequence_from_motif(w, Theta):
    sequence = []
    for j in range(w):
        probs = [Theta[i][j] for i in range(4)]
        letter = int(np.random.choice([1, 2, 3, 4], p=probs))
        sequence.append(letter)
    return sequence


def generate_sequence_from_background(w, ThetaB):
    sequence = []
    for _ in range(w):
        letter = int(np.random.choice([1, 2, 3, 4], p=ThetaB))
        sequence.append(letter)
    return sequence


def generate_data(w, k, alpha, Theta, ThetaB):
    data = []
    for _ in range(k):
        if np.random.rand() < alpha:
            sequence = generate_sequence_from_motif(w, Theta)
        else:
            sequence = generate_sequence_from_background(w, ThetaB)
        data.append(sequence)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', required=True, help='ścieżka do pliku z parametrami wejściowymi (.json)')
    parser.add_argument('--output', required=True, help='ścieżka do pliku wyjściowego (.json)')
    args = parser.parse_args()

    with open(args.params, 'r') as f:
        params = json.load(f)

    w = params['w']
    k = params['k']
    alpha = params['alpha']
    Theta = params['Theta']  # lista 4 x w
    ThetaB = params['ThetaB']  # lista długości 4

    data = generate_data(w, k, alpha, Theta, ThetaB)

    output = {
        'w': w,
        'k': k,
        'alpha': alpha,
        'data': data
    }

    with open(args.output, 'w') as f:
        json.dump(output, f)


if __name__ == '__main__':
    main()
