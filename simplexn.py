# Code adapted from https://github.com/samuelkolb/incal/releases
from z3 import *
import pac
import string
from incalp.problem import Domain, Problem
import math
import time
import numpy as np
from random import gauss
from sklearn import linear_model
import matplotlib.pyplot as plt

DIMENSION = 5
NUMBER_OF_MODELS = 50


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
    return theory, var_types, var_domains, variables


def generate_models(theory, n, z3_vars, var_domains):
    s = Solver()
    s.add(theory)
    domain_boundaries = And([And(z3_vars[v] > var_domains[v][0], z3_vars[v] < var_domains[v][1]) for v in z3_vars])
    s.add(domain_boundaries)
    models = []
    while s.check() == sat and len(models) < n:
        model = s.model()
        models.append(model)
        block = []
        # add constraint that blocks the same model from being returned again
        for d in model:
            c = d()
            block.append(Or(c <= model[d] - 0.1, c >= model[d] + 0.1))
        s.add(Or(block))
    return models


def make_rand_vector(dims):
    vec = [gauss(0, 1) for _ in range(dims)]
    mag = sum(x ** 2 for x in vec) ** .5
    return np.array([x / mag for x in vec])


def sigmoid(x):
    """
    :param x: number
    :return: steep sigmoid
    """
    return 1 / (1 + math.exp(-5 * x))


def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')


set_option(rational_to_decimal=True)

theory, z3_vars, domains, variables = simplexn(DIMENSION)

tic = time.perf_counter()
models = generate_models(theory, NUMBER_OF_MODELS, z3_vars, domains)
toc = time.perf_counter()

print(f"{len(models)} models generated in {toc - tic:0.1f} seconds.")

learner = pac.PACLearner(z3_vars, theory)

examples = [And([var() == model[var] for var in model]) for model in models]
#print(examples)

random_direction = make_rand_vector(DIMENSION)

points = np.array([[float(model[var].numerator_as_long()) / float(model[var].denominator_as_long()) for var in model]
                   for model in models])
#print(points)
sigm = np.vectorize(sigmoid)

fs = sigm(points.dot(random_direction))

# Create linear regression object
regr = linear_model.LinearRegression()
regr.fit(points, fs)

if DIMENSION == 1:
    # Plot outputs
    plt.scatter(points, fs,  color='black')
    abline(regr.coef_[0], regr.intercept_)
    plt.ylabel("Objective value")
    plt.xlabel("Points in simplex")
    plt.show()

estimated_f = sum([regr.coef_[i]*z3_vars[variables[i]] for i in range(DIMENSION)]) + regr.intercept_
print(f"f vector: {random_direction}")
print(f"Estimated coefficients: {regr.coef_}")

estimated_fs = []

for example in examples:
    f_val = 0.1
    rejects = True
    while rejects:
        f_val *= 2
        state, _ = learner.decide_pac([example], estimated_f <= f_val, 1)
        if state == "Accept":
            rejects = False
    U = f_val
    L = f_val/2
    accuracy = 3
    for i in range(accuracy):
        state, _ = learner.decide_pac([example], estimated_f <= (U+L)/2, 1)
        if state == "Accept":
            U = (U+L)/2
        else:
            L = (U+L)/2
    estimated_fs.append(U)

print(f"Real fs: {fs}")
print(f"PAC-estimated fs: {estimated_fs}")
print(f"Difference: {fs - estimated_fs}")
