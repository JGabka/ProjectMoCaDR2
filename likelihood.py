import numpy as np

def calc_lik_motif(seq, Theta):
    return np.prod([Theta[nuc - 1, j] for j, nuc in enumerate(seq)])

def calc_lik_back(seq, ThetaB):
    return np.prod([ThetaB[nuc - 1] for nuc in seq])