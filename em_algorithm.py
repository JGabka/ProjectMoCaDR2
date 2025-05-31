import numpy as np
from likelihood import calc_lik_motif, calc_lik_back
from initialization import initialize_params

def expectation_step(data, Theta, ThetaB, alpha):
    gammas = []
    for seq in data:
        p_motif = calc_lik_motif(seq, Theta)
        p_bg = calc_lik_back(seq, ThetaB)
        gamma = (alpha * p_motif) / (alpha * p_motif + (1 - alpha) * p_bg)
        gammas.append(gamma)
    return np.array(gammas)

def maximization_step(data, gammas, w):
    Theta = np.zeros((4, w))
    ThetaB = np.zeros(4)
    for i, seq in enumerate(data):
        gamma = gammas[i]
        for j, nuc in enumerate(seq):
            Theta[nuc - 1, j] += gamma
            ThetaB[nuc - 1] += (1 - gamma)
    Theta = np.maximum(Theta, 1e-8)
    Theta /= Theta.sum(axis=0)
    ThetaB = np.maximum(ThetaB, 1e-8)
    ThetaB /= ThetaB.sum()
    return Theta, ThetaB

def em_algorithm(data, alpha, w, estimate_alpha=False, max_iter=100, tol=1e-6):
    Theta, ThetaB = initialize_params(w)

    if estimate_alpha:
        alpha = 0.5

    for iteration in range(max_iter):
        gammas = expectation_step(data, Theta, ThetaB, alpha)
        Theta_new, ThetaB_new = maximization_step(data, gammas, w)

        if estimate_alpha:
            alpha_new = np.mean(gammas)
        else:
            alpha_new = alpha

        if (np.allclose(Theta, Theta_new, atol=tol) and
                np.allclose(ThetaB, ThetaB_new, atol=tol) and
                abs(alpha - alpha_new) < tol):
            break

        Theta, ThetaB, alpha = Theta_new, ThetaB_new, alpha_new

    return Theta, ThetaB, alpha
