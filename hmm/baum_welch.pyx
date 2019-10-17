import numpy as np

from hmm.logprob import ZERO, LogProb
from hmm.markov_chain import Transition, MarkovChain
from hmm.distributions import Multinomial, Gaussian, GaussianMixtureModel


def infer(hmm, sequence, fwd, bwd):
    cdef int T = sequence.shape[0]
    cdef int N = hmm.n_states    
    cdef double[:, :, :] zeta = np.ones((T - 1, N, N), dtype=np.double) * ZERO
    cdef int t, i, j
    for t in range(0, T - 1):
        norm = LogProb(ZERO)
        for i in range(0, N):
            for j in range(0, N):
                trans    = Transition(i, j)
                sample   = sequence[t + 1]

                logprob  = LogProb(fwd[t][i]) * hmm.transitions[trans] 
                logprob *= hmm.observations[j][sample]
                logprob *= LogProb(bwd[t + 1][j])

                norm         += logprob
                zeta[t][i][j] = logprob.prob
                
        for i in range(0, N):
            for j in range(0, N):
                zeta[t][i][j] -= norm.prob
    return np.asarray(zeta)


def markov(zetas, gammas, cls = MarkovChain):
    assert len(zetas) > 0 and len(zetas) == len(gammas)
    cdef int m = len(zetas)
    cdef int n = zetas[0].shape[1]
    cdef double[:, :] transitions = np.ones((n, n), dtype=np.double) * ZERO
    cdef int e, t, i, j
    for i in range(0, n):
        for j in range(0, n):
            scaler = LogProb(ZERO)
            for e in range(0, m):
                T = zetas[e].shape[0]
                for t in range(0, T):
                    logprob = LogProb(zetas[e][t,i,j]) + LogProb(transitions[i, j])
                    transitions[i,j] = logprob.prob
                    scaler += LogProb(gammas[e][t, i])
            prob = LogProb(transitions[i,j]) / scaler
            transitions[i, j] = prob.exp
    return cls.from_probs(transitions)


def discrete_obs(sequences, gammas, domain):
    assert len(gammas) > 0 and len(sequences) == len(gammas)
    cdef int m = len(gammas)
    cdef int n = gammas[0].shape[1]
    cdef multinomials = []
    cdef int i, k, e, t
    for i in range(0, n):
        multinomial  = {}     
        for k in domain:
            scaler = LogProb(ZERO)
            prob   = LogProb(ZERO)
            for e in range(0, m):
                T = gammas[e].shape[0]
                for t in range(0, T):
                    if sequences[e][t] == k:
                        prob += LogProb(gammas[e][t, i])
                    scaler += LogProb(gammas[e][t, i])
            multinomial[k] = (prob / scaler).exp
        observations = Multinomial(multinomial, domain)
        multinomials.append(observations)
    return multinomials

def continuous_tied_mixture(sequences, gammas, prev_obs, min_variance=1.0):
    assert len(gammas) > 0 and len(sequences) == len(gammas)
    cdef int m = len(gammas)
    cdef int n = gammas[0].shape[1]
    cdef int d = sequences[0].shape[1]
    cdef int k = len(prev_obs[0].probs)
    cdef int i, j, e, t, l
    mu       = np.zeros((k, d))
    sigma    = np.zeros((k, d))
    scaler_gaussians = np.zeros(k)
    probabilities = []        
    for i in range(0, n):
        scaler_cmp = LogProb(ZERO)
        cmp_prob   = []
        for j in range(0, k):
            prob_cmp = LogProb(ZERO) 
            for e in range(0, m):
                T = gammas[e].shape[0]
                for t in range(0, T):
                    g                    = LogProb(gammas[e][t, i]) * prev_obs[i].at(j, sequences[e][t])
                    weight               = g.exp
                    prob_cmp            += g
                    mu[j, :]            += sequences[e][t] * weight
                    scaler_gaussians[j] += weight
                    scaler_cmp          += g
            cmp_prob.append(prob_cmp)
        probabilities.append(np.array([(prob / scaler_cmp).exp for prob in cmp_prob]))

    for i in range(0, n):
        for j in range(0, k):
            mean = mu[j, :] / scaler_gaussians[j]
            for e in range(0, m):
                T = gammas[e].shape[0]
                for t in range(0, T):
                    g                 = LogProb(gammas[e][t, i]) * prev_obs[i].at(j, sequences[e][t])
                    weight            = g.exp
                    sigma[j, :]      += np.square(sequences[e][t] - mean) * weight
    
    for j in range(0, k):
        for l in range(0, d):
            sigma[j, l] = max(sigma[j,l] / scaler_gaussians[j], min_variance)

    observations = [GaussianMixtureModel(
        probabilities[i], 
        [Gaussian(mu[j, :] / scaler_gaussians[j], sigma[j, :]) 
        for j in range(0, k)
    ]) for i in range(0, n)]
    return observations

def continuous_mixture(sequences, gammas, prev_obs, min_variance=1.0):
    assert len(gammas) > 0 and len(sequences) == len(gammas)
    cdef int m = len(gammas)
    cdef int n = gammas[0].shape[1]
    cdef int d = sequences[0].shape[1]
    cdef int k = len(prev_obs[0].probs)
    cdef int i, j, e, t, l
    observations = []
    for i in range(0, n):
        scaler_cmp = LogProb(ZERO)
        cmp_prob   = []
        cmp_gauss  = []
        for j in range(0, k):
            mu       = np.zeros(d)
            sigma    = np.zeros(d) 
            prob_cmp = LogProb(ZERO) 
            scaler_gaussians = 0 
            for e in range(0, m):
                T = gammas[e].shape[0]
                for t in range(0, T):
                    g                 = LogProb(gammas[e][t, i]) * prev_obs[i].at(j, sequences[e][t])
                    weight            = g.exp
                    prob_cmp         += g
                    mu               += sequences[e][t] * weight
                    scaler_gaussians += weight
                    scaler_cmp       += g
            mu /= scaler_gaussians
            for e in range(0, m):
                T = gammas[e].shape[0]
                for t in range(0, T):
                    g                 = LogProb(gammas[e][t, i]) * prev_obs[i].at(j, sequences[e][t])
                    weight            = g.exp
                    sigma += np.square(sequences[e][t] - mu) * weight
            sigma /= scaler_gaussians
            for l in range(0, d):
                sigma[l] = max(min_variance, sigma[l])
            cmp_gauss.append(Gaussian(mu, sigma))
            cmp_prob.append(prob_cmp)
        probs = np.array([(prob / scaler_cmp).exp for prob in cmp_prob])
        observations.append(GaussianMixtureModel(probs, cmp_gauss))
    return observations

def continuous_obs(sequences, gammas, min_variance=1.0):
    assert len(gammas) > 0 and len(sequences) == len(gammas)
    cdef int m = len(gammas)
    cdef int n = gammas[0].shape[1]
    cdef int d = sequences[0].shape[1]        
    cdef observations = []
    cdef int i, e, t, j
    for i in range(0, n):
        mu     = np.zeros(d)
        sigma  = np.zeros(d) 
        scaler = 0 
        for e in range(0, m):
            T = gammas[e].shape[0]
            for t in range(0, T):
                weight = LogProb(gammas[e][t, i]).exp
                mu     += sequences[e][t] * weight
                scaler += weight
        mu /= scaler
        for e in range(0, m):
            T = gammas[e].shape[0]
            for t in range(0, T):
                weight = LogProb(gammas[e][t, i]).exp
                sigma += np.square(sequences[e][t] - mu) * weight
        sigma /= scaler
        for j in range(0, d):
            sigma[j] = max(min_variance, sigma[j])
        observations.append(Gaussian(mu, sigma))
    return observations