class HiddenMarkovModel:

    def __init__(self, transitions, observations):
        self.transitions = transitions
        self.observations = observations

    def n_states(self):
        self.transitions.n_states  