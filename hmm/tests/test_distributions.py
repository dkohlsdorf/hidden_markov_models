import unittest 
import numpy as np
from math import log
from functools import reduce

from hmm.distributions import Multinomial, Gaussian, GaussianMixtureModel
from hmm.logprob import LogProb, ZERO

class GaussianMixtureModelTest(unittest.TestCase):
    
    def test_mixture(self):
        gaussians = [Gaussian(np.zeros(3), np.ones(3)), Gaussian(np.ones(3) * 10, np.ones(3) * 20)]
        probs     = [0.5, 0.5]
        mixture   = GaussianMixtureModel(probs, gaussians)
        x         = np.ones(3) * 10
        self.assertAlmostEqual(
            mixture[x].prob, 
            log(0.5 * GaussianTest.normal(x) + 0.5 * GaussianTest.normal(x, 10, 20)), 
            delta=1e-8
        )

    def test_at(self):
        gaussians = [Gaussian(np.zeros(3), np.ones(3)), Gaussian(np.ones(3) * 10, np.ones(3) * 20)]
        probs     = [0.5, 0.5]
        mixture   = GaussianMixtureModel(probs, gaussians)
        x         = np.ones(3) * 10
        scaler    = 0.5 * GaussianTest.normal(x) + 0.5 * GaussianTest.normal(x, 10, 20)
        self.assertAlmostEqual(mixture.at(0, x).prob, log((0.5 * GaussianTest.normal(x)) / scaler ) , delta=1e-8)


class MultinomialTest(unittest.TestCase):

    def test_multinomial(self):
        multinomial = Multinomial({0:0.2, 1:0.8})
        self.assertEqual(multinomial.domain, set([0, 1]))

    def test_loglikelihood(self):
        multinomial = Multinomial({0:0.2, 1:0.8})
        self.assertAlmostEqual(multinomial[0].prob, log(0.2), delta=1e-8)
        self.assertAlmostEqual(multinomial[1].prob, log(0.8), delta=1e-8)        


class GaussianTest(unittest.TestCase):

    @classmethod
    def normal(cls, x, mu = 0, sigma = 1):
        scaler = 1.0 / np.sqrt(2.0 * np.pi * sigma)
        error  = np.exp(-0.5 * np.square((x - mu)) / sigma)
        return reduce(lambda x, y: x * y, scaler * error)
        
    def test_gaussian(self):
        gaussian = Gaussian(np.zeros(3), np.ones(3))
        ll_gaussian = gaussian[np.zeros(3)]
        ll_expected = LogProb.from_float(GaussianTest.normal(np.zeros(3)))
        self.assertAlmostEqual(ll_gaussian.prob, ll_expected.prob, delta=1e-8)
        gaussian    = Gaussian(np.ones(3) * 4, np.ones(3) * 15)
        ll_gaussian = gaussian[np.ones(3) * 2]
        ll_expected = LogProb.from_float(GaussianTest.normal(np.ones(3) * 2, 4, 15))
        self.assertAlmostEqual(ll_gaussian.prob, ll_expected.prob, delta=1e-8)
