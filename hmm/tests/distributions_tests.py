import unittest 
import numpy as np
from math import log
from functools import reduce

from hmm.distributions import Multinomial, Gaussian
from hmm.logprob import LogProb, ZERO


class MultinomialTest(unittest.TestCase):

    def test_multinomial(self):
        multinomial = Multinomial({0:0.2, 1:0.8})
        self.assertEqual(multinomial.domain, set([0, 1]))

    def test_loglikelihood(self):
        multinomial = Multinomial({0:0.2, 1:0.8})
        self.assertAlmostEqual(multinomial[0].prob, log(0.2), 1e-8)
        self.assertAlmostEqual(multinomial[1].prob, log(0.8), 1e-8)


class GaussianTest(unittest.TestCase):

    @classmethod
    def normal(cls, x):
        scaler = 1.0 / np.sqrt(2.0 * np.pi)
        error  = np.exp(np.square(-x) / 2.0 )
        return reduce(lambda x, y: x * y, scaler * error)
        
    def test_gaussian(self):
        gaussian = Gaussian(np.zeros(3), np.ones(3))
        ll_gaussian = gaussian[np.zeros(3)]
        ll_expected = LogProb.from_float(GaussianTest.normal(np.zeros(3)))
        self.assertAlmostEqual(ll_gaussian.prob, ll_expected.prob, delta=1e-8)