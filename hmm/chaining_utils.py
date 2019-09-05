from hmm.markov_chain import MarkovChain, Transition
from hmm.logprob import LogProb    


def combine_markov(chain1, chain2):
    offset   = chain1.n_states
    combined = MarkovChain()
    for transition, prob in chain1.transitions.items():
        combined[transition] = prob
    for transition, prob in chain2.transitions.items():
        i = transition.from_state
        if i >= 0:
            i += offset
        j = transition.to_state
        if j >= 0:
            j += offset
        offset_transition = Transition(i, j)
        combined[offset_transition] = prob
    return combined, offset


def combine_all(chains):
    combined = MarkovChain()
    offsets  = []  
    for chain in chains:        
        combined, offset = combine_markov(combined, chain)
        offsets.append(offset)    
    return combined, offsets + [combined.n_states]


def connect_models(combined, offsets, model2model):
    for (i,j) in model2model:
        connect_end   = offsets[i + 1] - 1
        connect_start = offsets[j]
        transition    = Transition(connect_end, connect_start)
        combined[transition] = LogProb.from_float(1.0)


def connect_hmms(hmms, model2model = []):
    pass