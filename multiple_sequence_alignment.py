import numpy as np
import matplotlib.pyplot as plt

from hmm.features.audio import *
from hmm.markov_chain import *
from hmm.logprob import * 

sequences = [
    cepstrum(spectrogram_from_file('data/whistle66.wav')),
    cepstrum(spectrogram_from_file('data/whistle71.wav'))
]

transitions = MarkovChain()
n_states = int(np.mean([x.shape[0] for x in sequences])) 
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
    