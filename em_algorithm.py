import numpy as np
from likelihood import calc_lik_motif, calc_lik_back
from initialization import initialize_params

def expectation_step(data, Theta, ThetaB, alpha):
    responsibilities = []
    for seq in data:
        motif_lik = calc_lik_motif(seq, Theta)
        bg_lik = calc_lik_back(seq, ThetaB)
        denom = alpha * motif_lik + (1 - alpha) * bg_lik
        gamma = (alpha * motif_lik) / denom
        responsibilities.append(gamma)
    return np.array(responsibilities)

def maximization_step(data, responsibilities, w):
    Theta = np.zeros((4, w))
    ThetaB = np.zeros(4)
    total_gamma = np.sum(responsibilities)
    total_1_gamma = len(data) - total_gamma

    for i, seq in enumerate(data):
        gamma = responsibilities[i]
        for pos, nuc in enumerate(seq):
            Theta[nuc - 1, pos] += gamma
            ThetaB[nuc - 1] += (1 - gamma)


    for j in range(w):
        Theta[:, j] /= np.sum(Theta[:, j])
    ThetaB /= np.sum(ThetaB)

    return Theta, ThetaB

def run_em(data, w, alpha, max_iter=100, tol=1e-6):
    Theta, ThetaB = initialize_params(w)
    for _ in range(max_iter):
        responsibilities = expectation_step(data, Theta, ThetaB, alpha)
        new_Theta, new_ThetaB = maximization_step(data, responsibilities, w)

        if np.allclose(Theta, new_Theta, atol=tol) and np.allclose(ThetaB, new_ThetaB, atol=tol):
            break
        Theta, ThetaB = new_Theta, new_ThetaB
    return Theta, ThetaB