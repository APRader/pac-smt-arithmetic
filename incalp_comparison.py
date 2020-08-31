# This code uses functions taken from https://github.com/samuelkolb/incal/releases
# All the modules from IncalP are placed in the folder incalp

import time
import numpy as np
from random import gauss
from sklearn import linear_model
from incalp.incalpsmt import LPLearner
from incalp.incremental_learner import RandomViolationsStrategy
import random
from incalp.lp_problems import simplexn, cuben, pollutionreduction, police
from incalp.smt_check import SmtChecker
import matplotlib.pyplot as plt
from z3 import *


def get_samples(problem, num_pos_samples, num_neg_samples):
    """
    Returns both positive and negative samples using random uniform sampling.
    :param problem: Problem object.
    :param num_pos_samples: Number of positive samples needed.
    :param num_neg_samples: Number of negative samples needed.
    :return: Lists of positive and negative samples.
    """
    true_samples = []
    false_samples = []

    while len(true_samples) < num_pos_samples or len(false_samples) < num_neg_samples:
        instance = dict()

        for variable in problem.domain.variables:
            lb, ub = problem.domain.var_domains[variable]
            instance[variable] = random.uniform(lb, ub)
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
    model = learner.learn(domain, samples, initial_indices)
    return model


def convert_to_smtlib(model, domain, objective_f, goal="maximise"):
    """
    Converts a PySMT model to an optimisation problem in SMTLIB format.
    :param model: A PySMT formula.
    :param domain: A Domain object.
    :param objective_f: The objective PySMT function.
    :param goal: Maximise or minimise.
    :return: A string containing the full optimisation SMTLIB problem.
    """
    if goal not in ("maximise", "minimise"):
        raise ValueError("Goal must either be 'maximise' or 'minimise'.")

    smtlib_model = model.to_smtlib()
    smtlib_f = objective_f.to_smtlib()

    # Creating the SMTLIB string and declaring the objective function
    smtlib_problem = "(declare-fun objective-f () Real)"
    # Adding variable declarations and domains
    for variable in domain.real_vars:
        smtlib_problem += f"(declare-const {variable} {domain.var_types.get(variable)})"\
                          f"(assert (>= {variable} {domain.var_domains.get(variable)[0]}))"\
                          f"(assert (<= {variable} {domain.var_domains.get(variable)[1]}))"
    # Adding hard and soft constraint
    smtlib_problem += f"(assert {smtlib_model})(assert (= objective-f {smtlib_f}))"
    if goal == "maximise":
        smtlib_problem += "(maximize objective-f)"
    else:
        smtlib_problem += "(minimize objective-f)"

    return smtlib_problem


def optimise_model(smtlib_problem):
    """
    Finding the optimal objective value function within a model.
    :param smtlib_problem: The entire optimisation problem in SMTLIB format.
    :return: The optimum objective value.
    """
    opt = Optimize()
    opt.from_string(smtlib_problem)
    opt.check()
    m = opt.model()
    for d in m.decls():
        if d.name() == "objective-f":
            r = m[d]
            return float(r.numerator_as_long()) / float(r.denominator_as_long())
    return None


def create_objective_function(problem, true_samples, dimensions):
    """
    Create a positive random objective function.
    :param problem: Problem object.
    :param true_samples: True samples.
    :param dimensions: Number of dimensions.
    :return: The objective function as a formula.
    """
    variables = problem.domain.variables
    if problem.name == "simplexn" or problem.name == "cuben":
        # We create a random objective function
        random_direction = make_rand_vector(dimensions)
        true_points = np.array([list(sample[0].values()) for sample in true_samples])
        fs = true_points.dot(random_direction)
        noisy_fs = fs + np.random.normal(0, 0.05, fs.shape)
        regr = linear_model.LinearRegression()
        regr.fit(true_points, noisy_fs)

        estimated_f = sum([regr.coef_[i].item() * problem.domain.get_symbol(variables[i]) for i in
                           range(dimensions)]) + regr.intercept_.item()
    elif problem.name == "pollution":
        coefficients = np.array([8, 10, 7, 6, 11, 9])
        variable_symbols = np.array([problem.domain.get_symbol(variable) for variable in variables])
        estimated_f = np.dot(coefficients, variable_symbols)
    elif problem.name == "police":
        coefficients = np.array([4, 4, 4, 4, 2])
        variable_symbols = np.array([problem.domain.get_symbol(variable) for variable in variables])
        estimated_f = np.dot(coefficients, variable_symbols)

    return estimated_f


def pac_learning(samples, query, validity):
    """
    Run Decide-PAC.
    :param samples: Positive samples.
    :param query: Query.
    :param validity: A number from 0 to 1 that represents the validity. Higher number means higher validity.
    :return: True if the samples are valid, False otherwise.
    """
    num_true_samples = 0
    for sample in samples:
        if SmtChecker(sample[0]).check(query):
            num_true_samples += 1
    return num_true_samples / len(samples) >= validity


def run_pac(samples, estimated_f, goal="maximise", validity=1, accuracy=64):
    """
    Run PAC to find optimal objective value.
    :param accuracy: Number of iterations of halving the bounded interval.
    :param samples: Positive samples.
    :param estimated_f: SMT formula for the estimated objective function.
    :param goal: Whether the goal is to maximise or minimise the objective function.
    :param validity: The ratio of samples that must be valid against the queries.
    :return: The estimated optimal value of the objective function.
    """
    lower = None
    upper = None

    if goal not in ("maximise", "minimise"):
        raise ValueError("Goal must either be 'maximise' or 'minimise'.")

    if goal == "minimise":
        # Minimisation is the same as maximisation with f of opposite sign
        estimated_f = -estimated_f

    if pac_learning(samples, 0 >= estimated_f, validity):
        if not pac_learning(samples, -1 >= estimated_f, validity):
            # optimal objective value is between -1 and 0, but we only set magnitudes
            lower = -1
            upper = 0
        else:
            # optimal objective value is less than -1, we find rough bounds using exponential search
            bound = -2
            while pac_learning(samples, bound >= estimated_f, validity):
                bound *= 2
            lower = bound
            upper = bound / 2
    else:
        if pac_learning(samples, 1 >= estimated_f, validity):
            # optimal objective value is between 0 and 1
            lower = 0
            upper = 1
        else:
            # optimal objective value is greater than 1, we find rough bounds using exponential search
            bound = 2
            while not pac_learning(samples, bound >= estimated_f, validity):
                bound *= 2
            lower = bound / 2
            upper = bound

    # find tight bounds for optimal objective value using binary search
    for i in range(accuracy):
        if pac_learning(samples, (lower + upper) / 2 >= estimated_f, validity):
            upper = (lower + upper) / 2
        else:
            lower = (lower + upper) / 2

    pac_estimated_f = (lower + upper) / 2

    if goal == "minimise":
        return -pac_estimated_f
    else:
        return pac_estimated_f


def make_rand_vector(dims):
    """
    Create a vector in a random dimension.
    :param dims: Number of dimensions of the vector.
    :return: The vector as a numpy array
    """
    vec = [gauss(0, 1) for _ in range(dims)]
    mag = sum(x ** 2 for x in vec) ** .5
    return np.array([x / mag for x in vec])


def main():
    if not len(sys.argv) == 2:
        print("The command requires exactly one argument. Please choose between simplexn, cuben, pollution or police.")
        sys.exit()
    else:
        problem_type = sys.argv[1]

    if problem_type == "simplexn" or problem_type == "cuben":
        all_dimensions = [2, 3, 4]
    elif problem_type == "pollution":
        # Pollution does not have different dimensions, so we just have this one
        all_dimensions = [2]
    elif problem_type == "police":
        all_dimensions = [2]
    else:
        print("Wrong argument given. Please provide either simplexn, cuben, pollution or police")
        sys.exit()

    sample_sizes = [50, 100, 200, 300, 400, 500]
    num_runs = 10
    log_path = "output/incalp_comparison_log.txt"
    incalp_runtimes = np.zeros((len(all_dimensions), len(sample_sizes), num_runs))
    pac_runtimes = np.zeros((len(all_dimensions), len(sample_sizes), num_runs))

    for run in range(num_runs):
        # Multiple independent runs
        print(f"Starting run {run + 1} out of {num_runs}.")

        for dimensions in all_dimensions:
            log_file = open(log_path, "a")
            log_file.write(f"DIMENSIONS: {dimensions}\n")
            log_file.close()

            if problem_type == "simplexn":
                problem = simplexn(dimensions)
                num_constraints = dimensions * (dimensions - 1) + 1
                pac_goal = "maximise"
            elif problem_type == "cuben":
                problem = cuben(dimensions)
                num_constraints = 2 * dimensions
                pac_goal = "maximise"
            elif problem_type == "pollution":
                problem = pollutionreduction()
                num_constraints = 3
                pac_goal = "minimise"
            elif problem_type == "police":
                problem = police()
                # 10 constraints, 3 of which are redundant
                num_constraints = 7
                pac_goal = "minimise"

            # We need at most max/2 samples
            max_samples = max(sample_sizes)
            all_true_samples, all_false_samples = get_samples(problem, max_samples // 2, max_samples // 2)

            for i, sample_size in enumerate(sample_sizes):
                log_file = open(log_path, "a")
                log_file.write(f"SAMPLES: {sample_size}\n")
                log_file.close()

                true_samples = all_true_samples[:int(sample_size / 2)]
                false_samples = all_false_samples[:int(sample_size / 2)]

                estimated_f = create_objective_function(problem, true_samples, dimensions)

                true_smtlib_problem = convert_to_smtlib(problem.theory, problem.domain, estimated_f, pac_goal)
                true_f = optimise_model(true_smtlib_problem)

                tic = time.perf_counter()
                model = run_incalp(problem.domain, true_samples + false_samples, num_constraints)
                toc = time.perf_counter()
                smtlib_problem = convert_to_smtlib(model, problem.domain, estimated_f, pac_goal)
                tuc = time.perf_counter()
                incalp_estimated_f = optimise_model(smtlib_problem)
                tac = time.perf_counter()
                incalp_runtimes[dimensions - 2, i, run] = toc - tic

                log_file = open(log_path, "a")
                log_file.write(
                    f"IncalP took {toc - tic:0.1f} + {tac - tuc:0.1f} seconds for a {problem.name} with {dimensions} dimensions "
                    f"and {sample_size} sample points.\n")

                log_file.close()

                tic = time.perf_counter()
                pac_estimated_f = run_pac(true_samples, estimated_f, pac_goal)
                toc = time.perf_counter()
                pac_runtimes[dimensions - 2, i, run] = toc - tic

                log_file = open(log_path, "a")
                log_file.write(
                    f"PAC took {toc - tic:0.1f} seconds for a {problem.name} with {dimensions} dimensions "
                    f"and {sample_size / 2} positive sample points.\n")
                log_file.close()

                print(f"IncalP estimated f: {incalp_estimated_f}")
                print(f"PAC estimated f: {pac_estimated_f}")
                print(f"True f: {true_f}\n")

    mean_incalp_runtimes = np.mean(incalp_runtimes, axis=2)
    mean_pac_runtimes = np.mean(pac_runtimes, axis=2)
    std_incalp_runtimes = np.std(incalp_runtimes, axis=2)
    std_pac_runtimes = np.std(pac_runtimes, axis=2)

    def create_plot(i, ax):
        y_incalp = mean_incalp_runtimes[i, :]
        y_pac = mean_pac_runtimes[i, :]
        y_err_incalp = std_incalp_runtimes[i, :]
        y_err_pac = std_pac_runtimes[i, :]
        ax.plot(sample_sizes, y_incalp, '--', label='IncalP(SMT)')
        ax.plot(sample_sizes, y_pac, label='PAC')
        ax.fill_between(sample_sizes, y_incalp - y_err_incalp, y_incalp + y_err_incalp, alpha=0.3)
        ax.fill_between(sample_sizes, y_pac - y_err_pac, y_pac + y_err_pac, alpha=0.3)
        ax.set_xlabel("Sample size")

    if problem_type in ("simplexn", "cuben"):
        fig, axs = plt.subplots(1, 3, sharey='all', constrained_layout=True, figsize=(12, 3))
        for i, ax in enumerate(axs):
            create_plot(i, ax)
            ax.title.set_text(f"n: {i + 2}")
        plt.setp(axs[0], ylabel='Time (s)')
        axs[0].legend()
        fig.suptitle(problem.name)
    else:
        fig, ax = plt.subplots(figsize=(4, 3))
        create_plot(0, ax)
        ax.set_ylabel('Time (s)')
        ax.legend()

    plot_file = f"output/{problem.name}-{time.time():.0f}.pdf"
    plt.savefig(plot_file)

    print(f"Finished! The plot can be found at {plot_file}.")


if __name__ == "__main__":
    main()
