import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

from hmm.features.audio import *
from hmm.markov_chain import *
from hmm.logprob import * 
from hmm.distributions import *
from hmm.hidden_markov_model import * 
from hmm.viterbi import *

import hmm.baum_welch as bw 
import hmm.fwd_bwd as infer

sequences = [
    cepstrum(spectrogram_from_file('data/whistle66.wav', 1024, 512)),
    cepstrum(spectrogram_from_file('data/whistle71.wav', 1024, 512))
]

n_states = 10
transitions = DenseMarkovChain(n_states)
per_state   = [int(max(x.shape[0] / n_states, 1)) for x in sequences]

transitions[Transition(START_STATE, 0)]           = LogProb.from_float(1.0)
transitions[Transition(n_states - 1, STOP_STATE)] = LogProb.from_float(1.0)
observations = []
for i in range(0, n_states):
    self_prob   = 0.25
    delete_prob = 0.25 / (n_states - i)
    match_prob  = 0.5
    for j in range(i, n_states):
        if i == j:
            transitions[Transition(i, j)] = LogProb.from_float(self_prob)
        elif i + 1 == j:
            transitions[Transition(i, j)] = LogProb.from_float(match_prob)
        else:
            transitions[Transition(i, j)] = LogProb.from_float(delete_prob)
    vectors = []
    for k in range(0, len(sequences)):
       for j in range(i * per_state[k], min((i + 1) * per_state[k], len(sequences[k]))):
           vectors.append(sequences[k][j, :])
    vectors = np.array(vectors)
    mu      = np.mean(vectors, axis=0)
    sigma   = np.var(vectors, axis=0) 
    observations.append(Gaussian(mu, sigma))

hmm = HiddenMarkovModel(transitions, observations)
pool = multiprocessing.Pool(processes=2)
for i in range(0, 10):
    inference    = pool.starmap(infer.infer, [(hmm, seq) for seq in sequences])
    gammas       = [gamma for gamma, _, _ in inference]
    zetas        = [bw.infer(hmm, sequences[i], inference[i][1], inference[i][2]) for i in range(0, len(inference))]    
    transitions  = bw.markov(zetas, gammas, DenseMarkovChain)
    transitions[Transition(START_STATE, 0)]           = LogProb.from_float(1.0)
    transitions[Transition(n_states - 1, STOP_STATE)] = LogProb.from_float(1.0)
    observations = bw.continuous_obs(sequences, gammas)
    hmm = HiddenMarkovModel(transitions, observations)
alignment = pool.starmap(viterbi, [(hmm, seq) for seq in sequences])
print(alignment)