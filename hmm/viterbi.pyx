import numpy as np
from hmm.markov_chain import  START_STATE, STOP_STATE, Transition
from hmm.logprob import ZERO, LogProb


def viterbi(hmm, double[:,:] sequence): 
    cdef int T = sequence.shape[0]
    cdef int N = hmm.n_states

    cdef double[:] sample = sequence[0, :]
    cdef double[:, :] dp  = np.zeros((T, N), dtype=np.double)

    cdef int[:, :] bp = np.zeros((T, N), dtype=np.int)
    cdef int[:]  path = np.zeros(T, dtype=np.int)
    
    cdef double ll  = 0.0
    cdef int argmax = 0

    cdef int t, i, j
    
    for i in range(0, N):
        init    = Transition(START_STATE, i)
        logprob = hmm.observations[i][sample] * hmm.transitions[init]
        dp[0, i] = logprob.prob

    for t in range(1, T):
        for i in range(1, N):
            max_ll = LogProb.from_float(0.0)
            for j in range(1, N):
                transition = Transition(j, i)
                logprob = LogProb(dp[t - 1, j]) * hmm.transitions[transition] 
                if logprob > max_ll:
                    max_ll = logprob
                    argmax = j
            sample   = sequence[t, :]
            logprob  = max_ll * hmm.observations[i][sample]
            dp[t, i] = logprob.prob
            bp[t, i] = argmax

    max_ll = LogProb.from_float(0.0)
    argmax = 0
    for i in range(0, N):
        end = Transition(i, STOP_STATE)
        logprob = hmm.transitions[end]
        dp[T - 1, i] *= logprob.prob
        if logprob > max_ll:
            max_ll = logprob
            argmax = j
    
    t = N - 2
    i = argmax
    while t > 0:
        path[t] = i
        i = bp[t, i]
        t -= 1
    return path, ll