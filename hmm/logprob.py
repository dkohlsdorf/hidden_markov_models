from math import log, isinf, exp


ZERO = float('-inf')


class LogProb:

    def __init__(self, prob):
        self.prob = prob

    @classmethod
    def from_float(cls, prob):
        if prob == 0.0:
            return cls(ZERO)
        return cls(log(prob))

    @property
    def is_zero(self):
        return isinf(self.prob) and self.prob < 0.0

    def __mul__(self, other):
        assert isinstance(other, LogProb)   
        logprob = self.prob + other.prob
        return LogProb(logprob)

    def __truediv__(self, other):
        assert isinstance(other, LogProb)  
        logprob = self.prob - other.prob
        return LogProb(logprob)

    def __add__(self, other):
        assert isinstance(other, LogProb)        
        if self.is_zero:
            return LogProb(other.prob)
        if other.is_zero:
            return LogProb(self.prob)
        if self.prob > other.prob:
            return LogProb(self.prob + log(1.0 + exp(other.prob - self.prob)))
        return LogProb(other.prob + log(1.0 + exp(self.prob - other.prob)))