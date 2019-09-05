import unittest
import numpy as np

from hmm.viterbi import viterbi
from hmm.logprob import LogProb

from hmm.tests.left_right_hmm import HMM

class ViterbiTest(unittest.TestCase):

    def test_viterbi(self):
        seq = np.array([0,0,0,1,1,1,0,0,0])
        path, ll = viterbi(HMM, seq)
        self.assertListEqual(list(path), [0,0,0,1,1,1,2,2,2])
        self.assertAlmostEqual(LogProb.from_float(0.001).prob, ll.prob, places=2)        

