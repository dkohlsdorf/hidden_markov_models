# TODO Move this to a notebook


import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

from collections import namedtuple

from hmm.features.conv_kmeans import *
from hmm.features.audio import *
from hmm.markov_chain import *
from hmm.logprob import * 
from hmm.distributions import *
from hmm.hidden_markov_model import * 
from hmm.viterbi import *
import hmm.baum_welch as bw 
import hmm.fwd_bwd as infer

pool = multiprocessing.Pool(processes=2)

MSATuple = namedtuple('MSATuple', 'start frame')

def pad(list, target):
    n = len(list)
    m = target - n
    padded = list    
    for _ in range(m):
        padded.append(None)
    return padded

def align_by_state(alignments, n_states, hmm):
    n = len(alignments)
    msa = [[[] for _ in range(n)] for _ in range(n_states)]
    for i in range(n):
        for s, frame in alignments[i]:
            msa[s][i].append(frame)
            
    max_states = [max([len(a) for a in s]) for s in msa]
    for a in range(0, n):
        for s in range(0, n_states):
            component = hmm.observations[s].component_all(msa[s][a])
            msa[s][a] = [component for _ in range(len(msa[s][a]))]
            msa[s][a] = pad(msa[s][a], max_states[s])
    return msa

print("\n\n")
print("#########################################")
print("# Multiple Sequence Alignment For Audio #")
print("# by Daniel Kohlsdorf                   #")
print("#########################################")

all_sequences = [
    spectrogram_from_file('data/whistle66.wav'),
    spectrogram_from_file('data/whistle71.wav'),
    spectrogram_from_file('data/burst15.wav'),
    spectrogram_from_file('data/burst2.wav'),
    spectrogram_from_file('data/burst3.wav'),
    spectrogram_from_file('data/burst4.wav')
]
sequences = all_sequences[0:2]

print("\tLearn Feature space")
km = ConvolutionalKMeans.from_dataset(sequences, k=32, win_t=10, win_d=10, max_iter=250)
sequences = [km[sequence] for sequence in sequences]

print("\tPlot Centroids")
k = len(km.centroids)
for i, centroid in enumerate(km.centroids):
    plt.subplot(1, k, i + 1)
    plt.imshow(np.reshape(centroid, (10, 10)).T)
plt.show()

print("\tPlot Sequences")
plt.subplot(2,1,1)
plt.imshow(sequences[0].T, cmap='gray')
plt.subplot(2,1,2)
plt.imshow(sequences[1].T, cmap='gray')
plt.show()

print("\tPlot Distances frame to frame")
n = len(sequences[0])
m = len(sequences[1])
features = np.zeros((n, m))
for i in range(0, n):
    for j in range(0, m):
        features[i][j] = np.sqrt(np.sum(np.square(sequences[0][i] - sequences[1][j])))
plt.imshow(features.T)
plt.show()

print("\tPlot Difference")
l = min(n, m)
diff = plt.imshow(np.square(all_sequences[0][:l] - all_sequences[1][:l]).T)
plt.show()

print("\tPlot Sequences")
plt.subplot(2,1,1)
plt.imshow(all_sequences[0].T, cmap='gray')
plt.subplot(2,1,2)
plt.imshow(all_sequences[1].T, cmap='gray')
plt.show()

print("\t Initialize Hidden Markov Model")
n_states = 8
transitions = DenseMarkovChain(n_states)
per_state   = [int(max(x.shape[0] / n_states, 1)) for x in sequences]

print("\t\t - Initial start and stop states")
for i in range(n_states):
    if i < n_states / 2:
        transitions[Transition(START_STATE, i)] = LogProb.from_float(1.0 / n_states / 2)
    if i >= n_states / 2:
        transitions[Transition(i, STOP_STATE)]  = LogProb.from_float(1.0 / n_states / 2)

print("\t\t - Initial Transitions")
for i in range(0, n_states):
    self_prob   = 0.1
    delete_prob = 0.1 / (n_states - i)
    match_prob  = 0.8
    for j in range(i, n_states):
        if i == j:
            transitions[Transition(i, j)] = LogProb.from_float(self_prob)
        elif i + 1 == j:
            transitions[Transition(i, j)] = LogProb.from_float(match_prob)
        else:
            transitions[Transition(i, j)] = LogProb.from_float(delete_prob)

print("\t\t - Build Gaussian Mixture")
vectors = []
for seq in sequences:
    for frame in seq:
        vectors.append(frame)
n_components = 4
gmm = GaussianMixtureModel.from_dataset(vectors, n_components)
observations = []
for i in range(0, n_states):    
    prob = np.zeros(n_components)
    for j in range(0, len(sequences)):
        batch = len(sequences[j]) // n_states
        for frame in sequences[j][i * batch:(i + 1) * batch]:
            prob[gmm.component(frame)] += 1
    prob /= sum(prob)
    observations.append(GaussianMixtureModel(prob, gmm.gaussians))
hmm  = HiddenMarkovModel(transitions, observations)
print(hmm.transitions)

print("\t Plot Components Map")
cmp_map = np.zeros((n_states, n_components))
for i, gmm in enumerate(hmm.observations):
    cmp_map[i, :] = np.array(gmm.probs)
plt.imshow(cmp_map)
plt.show()

print("\t EM: Baum Welch Estimation")
for i in range(0, 5):
    print("\t\t - iter: {}".format(i))
    inference    = pool.starmap(infer.infer, [(hmm, seq) for seq in sequences])
    gammas       = [gamma for gamma, _, _ in inference]
    zetas        = [bw.infer(hmm, sequences[i], inference[i][1], inference[i][2]) for i in range(0, len(inference))]    
    transitions  = bw.markov(zetas, gammas, DenseMarkovChain)
    observations = bw.continuous_tied_mixture(sequences, gammas, hmm.observations)
    for i in range(n_states):
        if i < n_states / 2:
            transitions[Transition(START_STATE, i)] = LogProb.from_float(1.0 / n_states / 2)
        if i >= n_states / 2:
            transitions[Transition(i, STOP_STATE)]  = LogProb.from_float(1.0 / n_states / 2)
    hmm = HiddenMarkovModel(transitions, observations)

print("\t Viterbi Alignemnt")    
print("\tTransitions")
print(hmm.transitions)
print("\t Plot Components Map")
cmp_map = np.zeros((n_states, n_components))
for i, gmm in enumerate(hmm.observations):
    cmp_map[i, :] = np.array(gmm.probs)
plt.imshow(cmp_map)
plt.show()

alignments = pool.starmap(viterbi, [(hmm, seq) for seq in sequences])
for alignment in alignments:
    print(alignment)

alignments = [[MSATuple(alignments[i][0][j], sequences[i][j, :]) for j in range(len(alignments[i][0]))] for i in range(len(alignments))]
alignments = align_by_state(alignments, n_states, hmm)
print("\n\n")

for i in range(0, len(sequences)):
    strg = ""
    for s in range(n_states):        
        for j in range(len(alignments[s][i])):
            if alignments[s][i][j] is not None:
                strg += "{}".format(alignments[s][i][j])
            else:
                strg += '_'
        strg += "|"
    print(strg)
print("\n\n")
