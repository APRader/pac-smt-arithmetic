# Code adapted from https://github.com/samuelkolb/incal/releases
import string
from pysmt.shortcuts import *
from incalp.problem import *
import math


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
        var_types[i] = REAL
        var_domains[i] = (0, 1)

    domain = Domain(variables, var_types, var_domains)
    s = [Symbol(s, REAL) for s in domain.variables]
    constraints.append(Plus(s) <= normalisation(2.7))

    for a in letters[:dimension]:
        count += 1
        x = domain.get_symbol(a)
        for b in letters[count:dimension]:
            y = domain.get_symbol(b)

            constraints.append(
                x * normalisation(1 / math.tan(math.pi / 12)) - y * normalisation(math.tan(math.pi / 12)) >= 0)
            constraints.append(y * normalisation((1 / math.tan(math.pi / 12))) - x * (math.tan(math.pi / 12)) >= 0)

    theory = And(i for i in constraints)
    return Problem(domain, theory, "simplexn")


problem = simplexn(5)
print(export_problem(problem))
