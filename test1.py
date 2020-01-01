from z3 import *

x, y, z = Reals('x y z')
s = Solver()
bg_knowledge = And(x > 0, x < 5)
example = And(y > 1, y < 3, z < 0)
query = x > y
sentence = And(bg_knowledge, example, Not(query))
s.add(sentence)
print(s.check())
print(s.model())

