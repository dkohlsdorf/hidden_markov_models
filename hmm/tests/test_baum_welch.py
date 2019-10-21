import unittest
import numpy as np

import matplotlib.pyplot as plt

import hmm.fwd_bwd    as infer
import hmm.baum_welch as bw
from hmm.hidden_markov_model import HiddenMarkovModel
from hmm.tests.left_right_hmm import HMM, HMM_CONT, HMM_MIX
from hmm.markov_chain import Transition, START_STATE, STOP_STATE, DenseMarkovChain
from hmm.logprob import ZERO, LogProb
from hmm.distributions import GaussianMixtureModel

class BaumWelchTests(unittest.TestCase):

    LEFT_RIGHT = [
        [0.7, 0.3, 0.0],
        [0.0, 0.7, 0.3],
        [0.0, 0.0, 1.0]
    ]

    @classmethod
    def dp2path(cls, dp):
        (T, N) = dp.shape
        path = []
        for t in range(0, T):
            max_ll    = ZERO
            max_state = 0
            for j in range(0, N):
                if dp[t, j] > max_ll:
                    max_ll = dp[t, j]
                    max_state = j
            path.append(max_state)
        return path

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
        self.assertEqual(len(obs), 3)
        self.assertGreater(obs[0][0], obs[0][1])
        self.assertGreater(obs[1][1], obs[1][0])
        self.assertGreater(obs[2][0], obs[2][1])

    def test_tied_mixture_obs(self):
        sequences = [
            np.array([
                np.zeros(1),
                np.zeros(1),
                np.zeros(1),
                np.ones(1),
                np.ones(1),
                np.ones(1),
                np.zeros(1),
                np.zeros(1),
                np.zeros(1)
            ]),
            np.array([
                np.ones(1) * 10,
                np.ones(1) * 10,
                np.ones(1) * 10,
                np.ones(1) * 100,
                np.ones(1) * 100,
                np.ones(1) * 100,
                np.ones(1) * 10,
                np.ones(1) * 10,
                np.ones(1) * 10
            ])
        ]
        hmm = HMM_MIX
        gmm = GaussianMixtureModel.from_dataset(np.vstack(sequences), 4)
        hmm.observations = [gmm for i in range(0, 3)]
        for _ in range(0, 50):
            inference    = [infer.infer(hmm, seq) for seq in sequences]
            gammas       = [gamma for gamma, _, _ in inference]
            obs          = bw.continuous_tied_mixture(sequences, gammas, hmm.observations)
            hmm.observations = obs
        self.assertEqual(round(gmm.gaussians[0].mean[0]), round(obs[0].gaussians[0].mean[0]))
        self.assertEqual(round(gmm.gaussians[1].mean[0]), round(obs[0].gaussians[1].mean[0]))
        self.assertEqual(round(gmm.gaussians[2].mean[0]), round(obs[0].gaussians[2].mean[0]))
        self.assertEqual(round(gmm.gaussians[3].mean[0]), round(obs[0].gaussians[3].mean[0]))


    def test_mixture_obs(self):
        sequences = [
            np.array([
                np.zeros(1),
                np.zeros(1),
                np.zeros(1),
                np.ones(1),
                np.ones(1),
                np.ones(1),
                np.zeros(1),
                np.zeros(1),
                np.zeros(1)
            ]),
            np.array([
                np.ones(1) * 10,
                np.ones(1) * 10,
                np.ones(1) * 10,
                np.ones(1) * 100,
                np.ones(1) * 100,
                np.ones(1) * 100,
                np.ones(1) * 10,
                np.ones(1) * 10,
                np.ones(1) * 10
            ])
        ]
        hmm = HMM_MIX
        for _ in range(0, 100):
            inference    = [infer.infer(hmm, seq) for seq in sequences]
            gammas       = [gamma for gamma, _, _ in inference]
            obs          = bw.continuous_mixture(sequences, gammas, hmm.observations)
            hmm.observations = obs
        self.assertAlmostEqual(0.5, obs[0].probs[0], delta=1e-1)
        self.assertAlmostEqual(0.5, obs[0].probs[1], delta=1e-1)
        self.assertAlmostEqual(0.5, obs[1].probs[0], delta=1e-1)
        self.assertAlmostEqual(0.5, obs[1].probs[1], delta=1e-1)
        self.assertAlmostEqual(0.5, obs[2].probs[0], delta=1e-1)
        self.assertAlmostEqual(0.5, obs[2].probs[1], delta=1e-1)
        self.assertEqual(10,  round(obs[0].gaussians[0].mean[0]))
        self.assertEqual(0,   round(obs[0].gaussians[1].mean[0]))
        self.assertEqual(100, round(obs[1].gaussians[0].mean[0]))
        self.assertEqual(1,   round(obs[1].gaussians[1].mean[0]))
        self.assertEqual(10,  round(obs[2].gaussians[0].mean[0]))
        self.assertEqual(0,   round(obs[2].gaussians[1].mean[0]))
        self.assertGreaterEqual(obs[0].gaussians[0].variance[0], 1.0)
        self.assertGreaterEqual(obs[0].gaussians[1].variance[0], 1.0)
        self.assertGreaterEqual(obs[1].gaussians[0].variance[0], 1.0)
        self.assertGreaterEqual(obs[1].gaussians[1].variance[0], 1.0)
        self.assertGreaterEqual(obs[2].gaussians[0].variance[0], 1.0)
        self.assertGreaterEqual(obs[2].gaussians[1].variance[0], 1.0)

    def test_continuous_obs(self):
        seq = np.array([
            np.zeros(1),
            np.zeros(1),
            np.zeros(1),
            np.ones(1),
            np.ones(1),
            np.ones(1),
            np.zeros(1),
            np.zeros(1),
            np.zeros(1)
        ])
        hmm = HMM_CONT
        for _ in range(0, 10):
            gamma, _, _ = infer.infer(HMM_CONT, seq)
            gammas    = [gamma]
            sequences = [seq]
            obs       = bw.continuous_obs(sequences, gammas)
            hmm.observations = obs
        self.assertEqual(round(obs[0].mean[0]), 0)
        self.assertEqual(round(obs[1].mean[0]), 1)
        self.assertEqual(round(obs[2].mean[0]), 0)
