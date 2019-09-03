import unittest
import numpy as np

from hmm.tests.left_right_hmm import HMM
from hmm.markov_chain import Transition
import hmm.fwd_bwd as infer
import hmm.baum_welch as bw

class BaumWelchTests(unittest.TestCase):

    LEFT_RIGHT = [
        [0.7, 0.3, 0.0],
        [0.0, 0.7, 0.3],
        [0.0, 0.0, 1.0]
    ]

    def test_markov(self):
        seq    = np.array([0,0,0,1,1,1,0,0,0])
        gamma, alpha, beta = infer.infer(HMM, seq) 
        gammas = [gamma]
        zetas  = [bw.infer(HMM, seq, alpha, beta)]    
        transitions = bw.markov(zetas, gammas)
        for i in range(0, 3):
            for j in range(0, 3):
                t = Transition(i, j)
                estimated = np.round(transitions[t].exp, 1)
                expected  = np.round(BaumWelchTests.LEFT_RIGHT[i][j], 1)
                self.assertAlmostEqual(estimated, expected, delta=1e-12)
    
    def test_discrete_obs(self):
        seq        = np.array([0,0,0,1,1,1,0,0,0])
        gamma, _,_ = infer.infer(HMM, seq) 

        gammas    = [gamma]
        sequences = [seq]
        domain    = HMM.observations[0].domain
        obs       = bw.discrete_obs(sequences, gammas, domain)
        self.assertEquals(len(obs), 3)
        print([x.event_probs for x in obs])
