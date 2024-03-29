import numpy as np
import io
from sklearn.mixture import GaussianMixture
from hmm.logprob import LogProb, ZERO


class GaussianMixtureModel:

    def __init__(self, probs, gaussians):
        assert len(probs) == len(gaussians)
        self.probs = probs
        self.gaussians = gaussians

    @classmethod
    def from_dataset(cls, dataset, k, random_state=0, max_iter=1024):
        gm = GaussianMixture(
            n_components=k, 
            covariance_type='diag', 
            tol = 1.0, 
            max_iter= max_iter, 
            random_state=random_state)
        gm.fit(dataset)
        variances = gm.covariances_
        means     = gm.means_
        probs     = gm.weights_ 
        gaussians = [Gaussian(means[i], variances[i]) for i in range(0, k)]
        return cls(probs, gaussians)

    def component_all(self, x):
        max_component = 0
        max_ll        = ZERO
        for i in range(0, self.k):
            ll = LogProb.from_float(1)            
            for sample in x:
                ll *= self.at(i, sample)
            if ll.prob > max_ll:        
                max_ll = ll.prob
                max_component = i
        return max_component

    def component(self, x):
        max_component = 0
        max_ll        = ZERO
        for i in range(0, self.k):
            ll = self.at(i, x)
            if ll.prob > max_ll:        
                max_ll = ll.prob
                max_component = i
        return max_component

    @property
    def k(self):
        return len(self.probs)

    def at(self, i, x):
        p = LogProb.from_float(self.probs[i]) * self.gaussians[i][x]
        scaler = self[x]
        return p / scaler

    def __getitem__(self, x):
        result = LogProb(ZERO)
        for i in range(self.k):
            result += LogProb.from_float(self.probs[i]) * self.gaussians[i][x]        
        return result


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
        error = -0.5 * np.square(x - self.mean) / self.variance
        return LogProb(self.scaler + np.sum(error))

    @property
    def std(self):
        return np.sqrt(self.variance)

    @property
    def dim(self):
        return len(self.variance)

    @property
    def scaler(self):
        return -((self.dim / 2) * np.log(2 * np.pi)) - np.sum(np.log(np.sqrt(self.variance)))