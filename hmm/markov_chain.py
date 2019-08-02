from hmm.logprob import LogProb, ZERO
from collections import namedtuple

Transition = namedtuple("Transition", "from_state to_state")


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