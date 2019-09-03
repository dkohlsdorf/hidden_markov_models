import unittest

import unittest
import numpy as np

from hmm.logprob import LogProb, ZERO
from hmm.fwd_bwd import fwd, bwd, infer

from hmm.tests.left_right_hmm import HMM

class FwdBwdTests(unittest.TestCase):

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

    def test_fwd(self):
        seq = np.array([0,0,0,1,1,1,0,0,0])
        dp = fwd(HMM, seq)
        self.assertListEqual(FwdBwdTests.dp2path(dp), [0, 0, 0, 1, 1, 1, 2, 2, 2])

    def test_bwd(self):
        seq = np.array([0,0,0,1,1,1,0,0,0])
        dp = bwd(HMM, seq)
        self.assertListEqual(FwdBwdTests.dp2path(dp), [0, 0, 1, 1, 1, 2, 2, 2, 2])

    def test_infer(self):
        seq = np.array([0,0,0,1,1,1,0,0,0])
        dp, _, _ = infer(HMM, seq)
        self.assertListEqual(FwdBwdTests.dp2path(dp), [0, 0, 0, 1, 1, 1, 2, 2, 2])
