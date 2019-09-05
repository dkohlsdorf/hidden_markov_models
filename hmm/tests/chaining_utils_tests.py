import unittest

from hmm.chaining_utils import *
from hmm.tests.left_right_hmm import *

class ChainUtilTest(unittest.TestCase):

    def test_combine_markov(self):
        chain1 = LEFT_RIGHT
        chain2 = LEFT_RIGHT
        combined, offset = combine_all([chain1, chain2])        
        print(combined)
        print(offset)
        connect_models(combined, offset, [(0, 1), (1, 0)])
        print(combined)