from z3 import *


def decide_pac(bg_knowledge, examples, query, validity):
    if 0 <= validity <= 1:
        epsilon = 1 - validity
    else:
        raise ValueError('validity must be between 0 and 1')
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
