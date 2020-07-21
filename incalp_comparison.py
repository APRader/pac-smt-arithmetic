# This code uses functions taken from https://github.com/samuelkolb/incal/releases
# All the modules from IncalP are placed in the folder incalp

import time
import numpy as np
from random import gauss
from sklearn import linear_model
from incalp.incalpsmt import LPLearner
from incalp.incremental_learner import RandomViolationsStrategy
import random
from incalp.lp_problems import simplexn, cuben
from incalp.smt_check import SmtChecker


def get_samples(problem, num_pos_samples, num_neg_samples):
    """
    Returns both positive and negative samples using random uniform sampling.
    :param problem: Problem object.
    :param num_pos_samples: Number of positive samples needed.
    :param num_neg_samples: Number of negative samples needed.
    :return: lists of positive and negative samples.
    """
    true_samples = []
    false_samples = []

    while len(true_samples) < num_pos_samples or len(false_samples) < num_neg_samples:
        instance = dict()

        for v in problem.domain.variables:
            lb, ub = problem.domain.var_domains[v]
            instance[v] = random.uniform(lb, ub)
        sample = (instance, SmtChecker(instance).check(problem.theory))

        if sample[1] and len(true_samples) < num_pos_samples:
            true_samples.append(sample)
        elif not sample[1] and len(false_samples) < num_neg_samples:
            false_samples.append(sample)

    return true_samples, false_samples


def run_incalp(domain, samples, num_half_spaces):
    """
    Runs IncalP with an LPLearner to learn a function given the samples incrementally.
    :param domain: The domain of each variable.
    :param samples: A list of positive and/or negative samples of the problem.
    :param num_half_spaces: The number of half-spaces the learned formula should contain.
    """
    # Starting with 20 random samples
    initial_indices = random.sample(range(len(samples)), 20)
    # Selecting 10 random violations after each round
    selection_strategy = RandomViolationsStrategy(10)
    learner = LPLearner(num_half_spaces, selection_strategy)
    learner.learn(domain, samples, initial_indices)


def create_objective_function(problem, true_samples, dimensions):
    """
    Create a positive random objective function.
    :param problem: Problem object.
    :param true_samples: True samples.
    :param dimensions: Number of dimensions.
    :return:
    """
    random_direction = np.abs(make_rand_vector(dimensions))
    true_points = np.array([list(sample[0].values()) for sample in true_samples])
    fs = true_points.dot(random_direction)
    noisy_fs = fs + np.random.normal(0, 0.1, fs.shape)
    regr = linear_model.LinearRegression()
    regr.fit(true_points, noisy_fs)

    variables = problem.domain.variables
    estimated_f = sum([regr.coef_[i].item() * problem.domain.get_symbol(variables[i]) for i in
                       range(dimensions)]) + regr.intercept_.item()
    return fs, noisy_fs, estimated_f


def pac_learning(samples, query):
    """
    Run Decide-PAC.
    :param samples: Positive samples.
    :param query: Query.
    :return: The ratio of true samples.
    """
    num_true_samples = 0
    for sample in samples:
        if SmtChecker(sample[0]).check(query):
            num_true_samples += 1
    return num_true_samples / len(samples)


def run_pac(samples, estimated_f):
    """
    Run PAC with binary search.
    :param samples: Positive samples.
    :param estimated_f: SMT formula for the estimated objective function.
    :return: The estimated highest value of the objective function.
    """
    f_val = 0.1
    rejects = True
    while rejects:
        f_val *= 2
        ratio_true = pac_learning(samples, estimated_f <= f_val)
        if ratio_true >= 1:
            rejects = False
    U = f_val
    L = f_val / 2
    accuracy = 1000
    for i in range(accuracy):
        ratio_true = pac_learning(samples, estimated_f <= (U + L) / 2)
        if ratio_true >= 1:
            U = (U + L) / 2
        else:
            L = (U + L) / 2
    pac_estimated_f = U
    return pac_estimated_f


def make_rand_vector(dims):
    """
    Return a vector in a random dimension.
    :param dims: Number of dimensions of the vector.
    :return:
    """
    vec = [gauss(0, 1) for _ in range(dims)]
    mag = sum(x ** 2 for x in vec) ** .5
    return np.array([x / mag for x in vec])


def main():
    incalp_runtimes = np.zeros((3, 6))
    pac_runtimes = np.zeros((3, 6))

    for dimensions in [2, 3, 4]:
        log_file = open("logs/incal_comparison_log.txt", "a")
        log_file.write(f"DIMENSIONS: {dimensions}\n")
        log_file.close()

        problem = simplexn(dimensions)
        # Number of constraints needed to represent SimplexN
        num_constraints = dimensions * (dimensions - 1) + 1

        # We need at most 500 samples
        all_true_samples, all_false_samples = get_samples(problem, 250, 250)

        for i, sample_size in enumerate([50, 100, 200, 300, 400, 500]):
            log_file = open("logs/incal_comparison_log.txt", "a")
            log_file.write(f"SAMPLES: {sample_size}\n")
            log_file.close()

            true_samples = all_true_samples[:int(sample_size / 2)]
            false_samples = all_false_samples[:int(sample_size / 2)]

            tic = time.perf_counter()
            run_incalp(problem.domain, true_samples + false_samples, num_constraints)
            toc = time.perf_counter()
            incalp_runtimes[dimensions - 2, i] = toc - tic

            log_file = open("logs/incal_comparison_log.txt", "a")
            log_file.write(f"IncalP took {toc - tic:0.1f} seconds for a {problem.name} with {dimensions} dimensions and {sample_size} sample points.\n")
            log_file.close()

            fs, noisy_fs, estimated_f = create_objective_function(problem, true_samples, dimensions)
            tic = time.perf_counter()
            pac_estimated_f = run_pac(true_samples, estimated_f)
            toc = time.perf_counter()
            pac_runtimes[dimensions - 2, i] = toc - tic

            log_file = open("logs/incal_comparison_log.txt", "a")
            log_file.write(f"PAC took {toc - tic:0.1f} seconds for a {problem.name} with {dimensions} dimensions and {sample_size / 2} positive sample points.\n")
            log_file.close()


if __name__ == "__main__":
    main()


"""
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
            block.append(Or(c <= model[d] - 0.01, c >= model[d] + 0.01))
        s.add(Or(block))
    return models
"""