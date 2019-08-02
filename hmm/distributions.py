from hmm.logprob import LogProb, ZERO


class Multinomial:

    def __init__(self, event_counts, domain = None, pseudo_counts = 0):
        if domain is None:
            self.domain = event_counts.keys()
        else:
            self.domain = domain
        self.event_counts = event_counts
        self.pseudo_counts = pseudo_counts

    def __getitem__(self, event):
        assert event in self.domain
        if event not in self.event_counts:
            prob = self.pseudo_counts / self.scaler
        else:
            prob = (self.event_counts[event] + self.pseudo_counts) / self.scaler
        return LogProb.from_float(prob)

    @property
    def scaler(self):
        z = 0.0
        for event in self.domain:
            if event in self.event_counts:
                z += self.event_counts[event]
            z += self.pseudo_counts
        return z