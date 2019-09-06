import numpy as np

from hmm.logprob import LogProb, ZERO
from collections import namedtuple

Transition = namedtuple("Transition", "from_state to_state")

START_STATE = -1
STOP_STATE  = -2

class MarkovChain:     

    def __init__(self):
        self.transitions = {}                                

    @classmethod
    def from_probs(cls, trans):
        n = len(trans)
        assert n > 0 and n == len(trans[0])        
        chain = MarkovChain()        
        for i in range(0, n):
            for j in range(0, n):
                chain[Transition(i, j)] = LogProb.from_float(trans[i][j])
        return chain

    def __setitem__(self, transition, prob): 
        assert isinstance(transition, Transition) and isinstance(prob, LogProb)  
        if not prob.is_zero:
            self.transitions[transition] = prob

    def __getitem__(self, transition):
        assert isinstance(transition, Transition)       
        if transition not in self.transitions:
            return LogProb(ZERO)
        return self.transitions[transition]

    @property
    def n_states(self):
        if len(self.transitions) == 0:
            return 0
        from_states = set([from_state for from_state, _ in self.transitions])
        to_states   = set([to_state for to_state, _ in self.transitions])
        states      = from_states | to_states
        return max(states) + 1

    def __repr__(self):
        n = self.n_states
        result = ""
        for i in range(0, n):
            t = Transition(START_STATE ,i)                
            result += "{}, ".format(np.round(self[t].exp, 1))
        result += '\n'
        for i in range(0, n):
            for j in range(0, n):
                t = Transition(i ,j)                
                result += "{}, ".format(np.round(self[t].exp, 1))
            result += '\n'
        for i in range(0, n):
            t = Transition(i ,STOP_STATE)                
            result += "{}, ".format(np.round(self[t].exp, 1))
        result += '\n'
        return result