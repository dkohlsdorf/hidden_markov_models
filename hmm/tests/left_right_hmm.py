import numpy as np

from hmm.hidden_markov_model import HiddenMarkovModel
from hmm.distributions import Multinomial, Gaussian
from hmm.markov_chain import MarkovChain, Transition, START_STATE, STOP_STATE
from hmm.logprob import LogProb


LEFT_RIGHT = MarkovChain.from_probs([
    [0.6, 0.4, 0.0],
    [0.0, 0.6, 0.4],
    [0.0, 0.0, 0.6]
])

LEFT_RIGHT[Transition(START_STATE, 0)] = LogProb(0.0)
LEFT_RIGHT[Transition(2, STOP_STATE)]  = LogProb(0.0)

OBS = [
    Multinomial({0:80, 1:20}),
    Multinomial({0:20, 1:80}),
    Multinomial({0:80, 1:20})
]

CONT = [
    Gaussian(np.zeros(1), np.ones(1)),
    Gaussian(np.ones(1),  np.ones(1)),
    Gaussian(np.zeros(1), np.ones(1))
]

HMM = HiddenMarkovModel(LEFT_RIGHT, OBS)
HMM_CONT = HiddenMarkovModel(LEFT_RIGHT, CONT)