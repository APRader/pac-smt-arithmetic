from z3 import *
import math


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


x, y= Reals('x y')
example_bg_knowledge = And(x > 0, x < 5)
example_examples = [And(y > 1, y < 3), And(y > -1, y < 2), And(y > -5, y < -3), And(y > 2, y < 7), And(y > 6, y < 8)]
example_query = x > y
print(decide_pac(example_bg_knowledge, example_examples, example_query, -0.1))
