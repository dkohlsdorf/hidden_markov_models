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


all_sequences = [
    mfcc_from_file('data/whistle66.wav'),
    mfcc_from_file('data/whistle71.wav'),
    mfcc_from_file('data/burst15.wav'),
    mfcc_from_file('data/burst2.wav'),
    mfcc_from_file('data/burst3.wav'),
    mfcc_from_file('data/burst4.wav')
]

sequences = all_sequences
n_states = 10
transitions = DenseMarkovChain(n_states)
per_state   = [int(max(x.shape[0] / n_states, 1)) for x in sequences]

for i in range(n_states):
    if i < n_states / 2:
        transitions[Transition(START_STATE, i)] = LogProb.from_float(1.0)
    if i >= n_states / 2:
        transitions[Transition(i, STOP_STATE)]  = LogProb.from_float(1.0)

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
for seq in sequences:
    for frame in seq:
        vectors.append(frame)
gmm = GaussianMixtureModel.from_dataset(vectors, 15)
observations = [gmm for i in range(n_states)]

hmm  = HiddenMarkovModel(transitions, observations)
pool = multiprocessing.Pool(processes=2)
for i in range(0, 10):
    inference    = pool.starmap(infer.infer, [(hmm, seq) for seq in sequences])
    gammas       = [gamma for gamma, _, _ in inference]
    zetas        = [bw.infer(hmm, sequences[i], inference[i][1], inference[i][2]) for i in range(0, len(inference))]    
    transitions  = bw.markov(zetas, gammas, DenseMarkovChain)
    transitions[Transition(START_STATE, 0)]           = LogProb.from_float(1.0)
    transitions[Transition(n_states - 1, STOP_STATE)] = LogProb.from_float(1.0)
    observations = bw.continuous_tied_mixture(sequences, gammas, hmm.observations)
    hmm = HiddenMarkovModel(transitions, observations)
alignments = pool.starmap(viterbi, [(hmm, seq) for seq in all_sequences])
alignments = [[(alignments[i][0][j], hmm.observations[alignments[i][0][j]].component(sequences[i][j, :])) for j in range(len(alignments[i][0]))] for i in range(len(alignments))]

for alignment in alignments:
    print(alignment)