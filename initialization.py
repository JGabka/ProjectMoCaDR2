import json
import numpy as np

def load_data(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    return data['data'], data['w'], data['k'], data['alpha']

def initialize_params(w):

    Theta = np.random.dirichlet([1, 1, 1, 1], size=w).T
    ThetaB = np.random.dirichlet([1, 1, 1, 1])
    return Theta, ThetaB
