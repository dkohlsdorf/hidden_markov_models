import numpy as np
from hmm.markov_chain import START_STATE, STOP_STATE, Transition
from hmm.logprob import ZERO, LogProb


def infer(hmm, sequence):
    cdef int T = sequence.shape[0]
    cdef int N = hmm.n_states    
    cdef double[:, :] gamma = np.ones((T, N), dtype=np.double) * ZERO

    alpha = fwd(hmm, sequence)
    beta  = bwd(hmm, sequence)

    for t in range(0, T):
        scaler = LogProb(ZERO)
        for i in range(0, N):
            logprob = LogProb(alpha[t, i]) * LogProb(beta[t, i])
            gamma[t, i] = logprob.prob
            scaler += logprob
        for i in range(0, N):
            logprob = LogProb(gamma[t,i]) / scaler
            gamma[t, i] = logprob.prob
    return np.asarray(gamma)



def bwd(hmm, sequence):
    cdef int T = sequence.shape[0]
    cdef int N = hmm.n_states    
    cdef double[:, :] dp = np.ones((T, N), dtype=np.double) * ZERO
    
    cdef int t, i, j
    for i in range(0, N):
        end          = Transition(i, STOP_STATE)
        logprob      = hmm.transitions[end]        
        dp[T - 1, i] = logprob.prob

    for t in range(T - 2, -1, -1):
        for i in range(0, N):
            pi = LogProb(ZERO)
            sample  = sequence[t + 1]
            for j in range(0, N):
                transition = Transition(i, j)
                pi += hmm.observations[j][sample] * hmm.transitions[transition] * LogProb(dp[t + 1, j])
            dp[t, i] = pi.prob
    
    for i in range(0, N):
        init     = Transition(START_STATE, i)
        logprob  = hmm.transitions[init] * LogProb(dp[0, i])
        dp[0, i] = logprob.prob

    return np.asarray(dp)


def fwd(hmm, sequence): 
    cdef int T = sequence.shape[0]
    cdef int N = hmm.n_states    
    cdef double[:, :] dp = np.ones((T, N), dtype=np.double) * ZERO
    
    cdef int t, i, j
    for i in range(0, N):
        init     = Transition(START_STATE, i)
        sample   = sequence[0]
        logprob  = hmm.observations[i][sample] * hmm.transitions[init]
        dp[0, i] = logprob.prob

    for t in range(1, T):
        for i in range(0, N):
            pi = LogProb(ZERO)
            for j in range(0, N):
                transition = Transition(j, i)       
                pi += LogProb(dp[t - 1, j]) * hmm.transitions[transition] 
            sample   = sequence[t]     
            pi = pi * hmm.observations[i][sample]
            dp[t, i] = pi.prob
    
    for i in range(0, N):
        end     = Transition(i, STOP_STATE)
        logprob = hmm.transitions[end] * LogProb(dp[T - 1, i])
        dp[T - 1, i] = logprob.prob 

    return np.asarray(dp)