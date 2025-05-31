
def calc_lik_motif(sequence, Theta):
    likelihood = 1.0
    for pos, nuc in enumerate(sequence):
        likelihood *= Theta[nuc - 1, pos]
    return likelihood

def calc_lik_back(sequence, ThetaB):
    likelihood = 1.0
    for nuc in sequence:
        likelihood *= ThetaB[nuc - 1]
    return likelihood