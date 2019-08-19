import unittest
import numpy as np


from hmm.viterbi import viterbi
from hmm.hidden_markov_model import HiddenMarkovModel
from hmm.distributions import Multinomial
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

HMM = HiddenMarkovModel(LEFT_RIGHT, OBS)

class ViterbiTests(unittest.TestCase):

    def test_viterbi(self):
        seq = np.array([0,0,0,1,1,1,0,0,0])
        path, ll = viterbi(HMM, seq)
        self.assertListEqual(list(path), [0,0,0,1,1,1,2,2,2])
        self.assertAlmostEqual(LogProb.from_float(0.001).prob, ll.prob, places=2)        

