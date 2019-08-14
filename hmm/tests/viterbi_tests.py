import unittest
import numpy as np

from hmm.viterbi import viterbi
from hmm.hidden_markov_model import HiddenMarkovModel
from hmm.distributions import Multinomial

LEFT_RIGHT = [
    [0.6, 0.4, 0.0],
    [0.0, 0.6, 0.4],
    [0.0, 0.0, 0.6]
]

OBS = [
    Multinomial({0:20, 1:80}),
    Multinomial({0:80, 1:20}),
    Multinomial({0:20, 1:80})
]

HMM = HiddenMarkovModel(LEFT_RIGHT, OBS)

class ViterbiTests(unittest.TestCase):

    def test_viterbi(self):
        seq = [0,0,0,1,1,1,0,0,0]
        print(viterbi(HMM, seq))
        

