import numpy as np
import matplotlib.pyplot as plt

from hmm.features.audio import *
from hmm.markov_chain import *
from hmm.logprob import * 
from hmm.distributions import *
from hmm.hidden_markov_model import * 
from hmm.viterbi import *

import hmm.baum_welch as bw 
import hmm.fwd_bwd as infer

sequences = [
    cepstrum(spectrogram_from_file('data/whistle66.wav')),
    cepstrum(spectrogram_from_file('data/whistle71.wav'))
]

transitions = MarkovChain()
n_states  = int(np.mean([x.shape[0] for x in sequences])) // 60
print(n_states)
per_state = [int(max(x.shape[0] / n_states, 1)) for x in sequences]

transitions[Transition(START_STATE, 0)]           = LogProb.from_float(1.0)
transitions[Transition(n_states - 1, STOP_STATE)] = LogProb.from_float(1.0)
observations = []
for i in range(0, n_states):
    self_prob   = 0.05
    delete_prob = 0.05 / (n_states - i)
    match_prob  = 0.9
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

print(transitions)
hmm = HiddenMarkovModel(transitions, observations)
for i in range(0, 10):
    inference    = [infer.infer(hmm, seq) for seq in sequences]
    gammas       = [gamma for gamma, _, _ in inference]
    zetas        = [bw.infer(hmm, sequences[i], inference[i][1], inference[i][2]) for i in range(0, len(inference))]    
    transitions  = bw.markov(zetas, gammas)
    transitions[Transition(START_STATE, 0)]           = LogProb.from_float(1.0)
    transitions[Transition(n_states - 1, STOP_STATE)] = LogProb.from_float(1.0)
    observations = bw.continuous_obs(sequences, gammas)
    hmm = HiddenMarkovModel(transitions, observations)

alignment = [viterbi(hmm, seq) for seq in sequences]
print(alignment)