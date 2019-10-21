import unittest

from hmm.chaining_utils import *
from hmm.tests.left_right_hmm import *

class ChainUtilTest(unittest.TestCase):

    @classmethod
    def expected_transitions(cls):
        combined = MarkovChain.from_probs([
            [0.6, 0.4, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.6, 0.4, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.6, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.6, 0.4, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.6, 0.4],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.6],
        ])
        combined[Transition(START_STATE, 0)] = LogProb.from_float(1.0)
        combined[Transition(START_STATE, 3)] = LogProb.from_float(1.0)
        combined[Transition(2, STOP_STATE)]  = LogProb.from_float(1.0)
        combined[Transition(5, STOP_STATE)]  = LogProb.from_float(1.0)
        return combined

    def test_combine_markov(self):
        chain1 = LEFT_RIGHT
        chain2 = LEFT_RIGHT
        combined, offset = combine_markov(chain1, chain2)       
        expected = ChainUtilTest.expected_transitions()
        self.assertEqual(offset, 3) 
        self.assertEqual(str(combined), str(expected))

    def test_combine_all(self):
        chain1 = LEFT_RIGHT
        chain2 = LEFT_RIGHT
        combined, offset = combine_all([chain1, chain2])      
        expected = ChainUtilTest.expected_transitions()
        self.assertListEqual(offset, [0, 3, 6]) 
        self.assertEqual(str(combined), str(expected))

    def test_connect_models(self):
        chain1 = LEFT_RIGHT
        chain2 = LEFT_RIGHT
        combined, offset = combine_all([chain1, chain2])      
        connect_models(combined, offset, [(0, 1), (1, 0)])
        self.assertEqual(combined[Transition(5, 0)].prob, LogProb.from_float(1.0).prob)
        self.assertEqual(combined[Transition(2, 3)].prob, LogProb.from_float(1.0).prob)
