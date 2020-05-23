from z3 import *
import itertools
import paclearner

A, B, C = Bools('A B C')
variables = [A, B, C]
Psi = [A, Or(B, Not(C)), Or(Not(A), Not(B), Not(C))]  # all constraints
cs = [0, 0, 1]  # hard constraints
ws = [1, 0.5, 0]  # weights of soft constraints
n = len(Psi)
assert (n == len(cs))
assert (n == len(ws))

examples = list(map(list, itertools.product([False, True], repeat=n)))
true_fs = []
true_cs = []
estimated_fs = []
weight_vars = [Real(f"w{i}") for i in range(n)]
kb = And([And(weight_var >= -1, weight_var <= 1) for weight_var in weight_vars])
f = Sum([weight_vars[i]*Psi[i] for i in range(n)])

for example in examples:
    w_star = 0
    c_star = True
    learner = paclearner.PACLearner(weight_vars, kb)
    s = Solver()
    s.add(A == example[0], B == example[1], C == example[2])
    s.push()
    for i in range(n):
        s.add(Psi[i])
        if s.check() == sat:
            w_star += ws[i]
            if cs[i] == 1:
                c_star = c_star and True
        else:
            if cs[i] == 1:
                c_star = c_star and False
        s.pop()
        s.push()
    true_fs.append(w_star)
    true_cs.append(c_star)
    assignments = And([variables[i] == example[i] for i in range(n)])
    state, _ = learner.decide_pac([assignments], f <= 1.5, 1)
    estimated_fs.append(state)


print(f"Constraints: {Psi}")
print(f"Weights: {ws}")
print(f"Hard constraints: {cs}")
print(f"Assignments: {examples}")
print(f"Their hard constraints: {true_cs}")
print(f"Their objective function values: {true_fs}")
print(f"Estimated objective function values: {estimated_fs}")
