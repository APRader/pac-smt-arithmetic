# Code adapted from https://github.com/samuelkolb/incal/releases
from z3 import *
import pac
import string
from incalp.problem import Domain, Problem
import math
import time


def simplexn(dimension):
    count = 0
    constraints = []

    def normalisation(x):
        return (x + 1) / ((2 + 2.7) - 1)

    variables = []
    var_types = {}
    var_domains = {}

    letters = list(string.ascii_lowercase)

    for i in letters[:dimension]:
        variables.append(i)
        var_types[i] = Real(i)
        var_domains[i] = (0, 1)

    s = [var_types[i] for i in variables]
    constraints.append(sum(s) <= normalisation(2.7))

    for a in letters[:dimension]:
        count += 1
        x = var_types[a]
        for b in letters[count:dimension]:
            y = var_types[b]

            constraints.append(
                x * normalisation(1 / math.tan(math.pi / 12)) - y * normalisation(math.tan(math.pi / 12)) >= 0)
            constraints.append(y * normalisation((1 / math.tan(math.pi / 12))) - x * (math.tan(math.pi / 12)) >= 0)

    theory = And(constraints)
    return theory, var_types, var_domains


def generate_models(theory, n):
    s = Solver()
    s.add(theory)
    models = []
    while s.check() == sat and len(models) < n:
        model = s.model()
        models.append(model)
        block = []
        # add constraint that blocks the same model from being returned again
        for d in model:
            c = d()
            block.append(Or(c <= model[d] - 0.01, c >= model[d] + 0.01))
        s.add(Or(block))
    return models


set_option(rational_to_decimal=True)

theory, z3_vars, domain = simplexn(5)

print(f"Theory: {theory}")

tic = time.perf_counter()
models = generate_models(theory, 10)
toc = time.perf_counter()

print(f"{len(models)} models generated in {toc - tic:0.1f} seconds.")

learner = pac.PACLearner(z3_vars, theory)

examples = [And([var() == model[var] for var in model]) for model in models]

