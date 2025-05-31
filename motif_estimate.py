import json
import argparse
from em_algorithm import em_algorithm
from initialization import load_data


def save_estimates(output_file, Theta, ThetaB):
    output = {
        "Theta": Theta.tolist(),
        "ThetaB": ThetaB.tolist()
    }
    with open(output_file, 'w') as f:
        json.dump(output, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Plik wejściowy z danymi .json')
    parser.add_argument('--output', required=True, help='Plik wyjściowy do zapisu estymacji .json')
    parser.add_argument('--estimate-alpha', default='no', help='Czy szacować alpha? (no = obowiązkowa wersja)')
    args = parser.parse_args()

    data, w, k, alpha = load_data(args.input)
    if args.estimate_alpha.lower() == 'yes':
        raise NotImplementedError("Bonusowa wersja z estymacją alpha jeszcze niezaimplementowana.")

    Theta, ThetaB = em_algorithm(data, w, alpha)
    save_estimates(args.output, Theta, ThetaB)

if __name__ == "__main__":
    main()