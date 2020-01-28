from z3 import *
import math


def decide_pac(bg_knowledge, examples, query, validity):
    if 0 <= validity <= 1:
        epsilon = 1 - validity
    else:
        raise ValueError('Validity must be between 0 and 1.')
    b = math.floor(epsilon * len(examples))
    failed = 0
    s = Solver()
    s.add(And(bg_knowledge, Not(query)))
    s.push()
    for example in examples:
        s.add(example)
        # if sat, then the entailment is rejected
        if s.check() == sat:
            failed += 1
            if failed > b:
                return 'Reject'
        s.pop()
        s.push()
    return 'Accept'


# returns number of samples needed to guarantee required confidence and gamma
def sample_size(confidence, gamma):
    if 0 < confidence <= 1:
        delta = 1 - confidence
    else:
        raise ValueError('Confidence must be >0 and <=1')
    if 0 < gamma <= 1:
        return math.ceil(1 / (2 * gamma * gamma) * math.log(1 / delta))
    else:
        raise ValueError('Gamma must be >0 and <=1')

