import numpy as np


from hmm.logprob import LogProb, ZERO


class Multinomial:

    def __init__(self, event_probs, domain = None):
        if domain is None:
            self.domain = event_probs.keys()
        else:
            self.domain = domain
        self.event_probs = event_probs

    def __getitem__(self, event):
        assert event in self.domain
        if event not in self.event_probs:
            prob = LogProb(ZERO)
        else:
            prob = self.event_probs[event]
        return LogProb.from_float(prob)        


class Gaussian:

    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance

    def __getitem__(self, x):
        error = np.square(x - self.mean) / self.variance
        return LogProb(np.sum(self.scaler + error))

    @property
    def std(self):
        return np.sqrt(self.variance)

    @property
    def scaler(self):
        return -0.5 * np.log(2 * np.pi * self.std)