from z3 import *
import pac

x, y = Reals('x y')
example_bg_knowledge = And(x > 0, x < 5)
example_examples = [And(y > 1, y < 3), And(y > -1, y < 2), And(y > -5, y < -3), And(y > 2, y < 7), And(y > 6, y < 8)]
example_query = x > y
print(pac.decide_pac(example_bg_knowledge, example_examples, example_query, 0.1))
