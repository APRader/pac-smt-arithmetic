# This code uses functions taken from https://github.com/samuelkolb/incal/releases
# All the modules from IncalP are placed in the folder incalp

import time
import numpy as np
from incalp.incalpsmt import LPLearner
from incalp.incremental_learner import RandomViolationsStrategy
import random
from incalp.lp_problems import simplexn, cuben, pollutionreduction, police
from incalp.smt_check import SmtChecker
import matplotlib.pyplot as plt
from z3 import Optimize
from pysmt.shortcuts import And, is_sat
import argparse
import string

SAMPLE_SIZES = [50, 100, 200, 300, 400, 500]
NUM_RUNS = 10
# SAMPLE_SIZES = [50, 100]
# NUM_RUNS = 2


def get_samples(problem, num_pos_samples, num_neg_samples):
    """
    Create both positive and negative samples using random uniform sampling.
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
    Run IncalP with an LPLearner to learn a model given the samples incrementally.
    :param domain: The domain of each variable.
    :param samples: A list of positive and/or negative samples of the problem.
    :param num_half_spaces: The number of half-spaces the learned model should contain.
    :return: The learned model.
    """
    # Starting with 20 random samples
    initial_indices = random.sample(range(len(samples)), 20)
    # Selecting 10 random violations after each round
    selection_strategy = RandomViolationsStrategy(10)
    learner = LPLearner(num_half_spaces, selection_strategy)
    return learner.learn(domain, samples, initial_indices)


def convert_to_smtlib(model, domain, objective_f, goal="maximise"):
    """
    Convert a PySMT model to an optimisation problem in SMTLIB format.
    :param model: A PySMT formula.
    :param domain: A domain object.
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
        smtlib_problem += f"(declare-const {variable} {domain.var_types[variable]})" \
                          f"(assert (>= {variable} {domain.var_domains[variable][0]}))" \
                          f"(assert (<= {variable} {domain.var_domains[variable][1]}))"
    # Adding hard and soft constraint
    smtlib_problem += f"(assert {smtlib_model})(assert (= objective-f {smtlib_f}))"
    if goal == "maximise":
        smtlib_problem += "(maximize objective-f)"
    else:
        smtlib_problem += "(minimize objective-f)"

    return smtlib_problem


def optimise_model(smtlib_problem):
    """
    Find the optimal objective function value within a model.
    :param smtlib_problem: The entire optimisation problem in SMTLIB format.
    :return: The optimum objective value.
    """
    opt = Optimize()
    opt.from_string(smtlib_problem)
    opt.check()
    m = opt.model()
    for d in m.decls():
        # It is assumed that the objective value constant is called objective-f
        if d.name() == "objective-f":
            # Getting the optimal value of the objective function
            r = m[d]
            # Convert Z3 variable into Python float
            return float(r.numerator_as_long()) / float(r.denominator_as_long())


def create_objective_function(problem):
    """
    Create the objective function corresponding to the problem.
    :rtype: FNode
    :param problem: Problem object.
    :return: The objective function as a PySMT formula.
    """
    variables = problem.domain.variables

    if problem.name in ("simplexn", "cuben"):
        # Creating a random linear objective function, since these problems don't have a canonical one
        coefficients = np.random.uniform(-1, 1, (len(variables) + 1))
    elif problem.name == "pollution":
        coefficients = np.array([8, 10, 7, 6, 11, 9, 0])
    elif problem.name == "police":
        coefficients = np.array([4, 4, 4, 4, 2, 0])
    else:
        raise ValueError("Problem must be either simplexn, cuben, pollution or police.")

    variable_symbols = np.array([problem.domain.get_symbol(variable) for variable in variables] + [1])
    return np.dot(coefficients, variable_symbols)


def create_intervals(domain, samples, width):
    """
    Turn assignments into intervals of given width.
    :param domain: Domain object.
    :param samples: List of samples.
    :param width: Width of interval.
    :return: List of PySMT formulas of each sample as intervals centered at the original assignment.
    """
    return [And([And(domain.get_symbol(variable) >= sample[0][variable] - width / 2,
                     domain.get_symbol(variable) <= sample[0][variable] + width / 2) for variable in sample[0]])
            for sample in samples]


def pac_learning(samples, query, validity, intervals=False):
    """
    Run DecidePAC to determine whether the samples satisfy the query.
    :param samples: Sample assignments or intervals.
    :param query: Query PySMT formula.
    :param validity: A number from 0 to 1 that represents the validity. Higher number means higher validity.
    :param intervals: Whether the samples are intervals rather than assignments.
    :return: True if the samples are valid, False otherwise.
    """
    num_true_samples = 0

    if intervals:
        for sample in samples:
            # Intervals are already PySMT formulas, so normal SAT checking can be used
            if is_sat(And(query, sample)):
                num_true_samples += 1
    else:
        for sample in samples:
            # Assignments have a special format that can be read by the check function from the IncalP code
            if SmtChecker(sample[0]).check(query):
                num_true_samples += 1

    return num_true_samples / len(samples) >= validity


def run_pac(samples, objective_f, goal="maximise", validity=1, accuracy=64):
    """
    Run OptimisePAC to find optimal objective value.
    :param accuracy: Number of iterations of halving the bounded interval.
    :param samples: Positive samples.
    :param objective_f: SMT formula for the objective function we want to optimise.
    :param goal: Whether the goal is to maximise or minimise the objective function.
    :param validity: The ratio of samples that must be valid against the queries.
    :return: The estimated optimal value of the objective function.
    """
    if goal not in ("maximise", "minimise"):
        raise ValueError("Goal must either be 'maximise' or 'minimise'.")

    if goal == "minimise":
        # Minimisation is the same as maximisation with objective function of opposite sign
        objective_f = -objective_f

    if pac_learning(samples, 0 >= objective_f, validity):
        # Optimal objective value is negative or 0
        if not pac_learning(samples, -1 >= objective_f, validity):
            lower = -1
            upper = 0
        else:
            # Optimal objective value is less than -1, finding rough bounds using exponential search
            bound = -2
            while pac_learning(samples, bound >= objective_f, validity):
                bound *= 2
            lower = bound
            upper = bound / 2
    else:
        # Optimal objective value is positive
        if pac_learning(samples, 1 >= objective_f, validity):
            lower = 0
            upper = 1
        else:
            # Optimal objective value is greater than 1, finding rough bounds using exponential search
            bound = 2
            while not pac_learning(samples, bound >= objective_f, validity):
                bound *= 2
            lower = bound / 2
            upper = bound

    # Finding tight bounds for optimal objective value using binary search
    for i in range(accuracy):
        if pac_learning(samples, (lower + upper) / 2 >= objective_f, validity):
            upper = (lower + upper) / 2
        else:
            lower = (lower + upper) / 2

    pac_estimated_f = (lower + upper) / 2

    if goal == "minimise":
        # Flipping sign back again
        return -pac_estimated_f
    else:
        return pac_estimated_f


def create_plot(ax, x_values, x_label, y1_values, y2_values, y1_error, y2_error, y1_label, y2_label):
    """
    Creates line graph for two sets of values.
    :param ax: Axis object.
    :param x_values: X values.
    :param x_label: Label for x axis.
    :param y1_values: Y values for first set of data.
    :param y2_values: Y values for second set of data.
    :param y1_error: Standard deviation for first set of data.
    :param y2_error: Standard deviation for second set of data.
    :param y1_label: Label for first set of data.
    :param y2_label: Label for second set of data.
    """
    ax.plot(x_values, y1_values, '--', label=y1_label)
    ax.plot(x_values, y2_values, label=y2_label)
    ax.fill_between(x_values, y1_values - y1_error, y1_values + y1_error, alpha=0.3)
    ax.fill_between(x_values, y2_values - y2_error, y2_values + y2_error, alpha=0.3)
    ax.set_xlabel(x_label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("problem", choices=["simplexn", "cuben", "pollution", "police"],
                        help="choose the type of problem to run the experiments on")
    parser.add_argument("-s", "--seed", type=int, help="set the seed for the random number generator")
    parser.add_argument("-v", "--verbose", action="store_true", help="turn on verbose mode")
    args = parser.parse_args()

    problem_type = args.problem
    verbose = args.verbose
    all_dimensions = None
    optimisation_goal = None
    problem = None

    if problem_type in ("simplexn", "cuben"):
        all_dimensions = [2, 3, 4]
        optimisation_goal = "maximise"
    elif problem_type == "pollution":
        all_dimensions = [3]
        optimisation_goal = "minimise"
    elif problem_type == "police":
        all_dimensions = [5]
        optimisation_goal = "minimise"

    # Creating a random log file name before seed is set
    characters = string.ascii_letters + string.digits
    random_string = ''.join((random.choice(characters) for _ in range(5)))
    log_path = f"output/{random_string}_incalp_comparison_log_{problem_type}.txt"
    print(f"Log file: {log_path}")

    # Setting the seed for random number generators
    seed = args.seed
    if seed:
        random.seed(seed)
        np.random.seed(seed)

    # Allocating arrays to store the running times
    incalp_runtimes = np.zeros((len(all_dimensions), len(SAMPLE_SIZES), NUM_RUNS))
    pac_runtimes = np.zeros((len(all_dimensions), len(SAMPLE_SIZES), NUM_RUNS))

    for run in range(NUM_RUNS):
        print(f"Starting run {run + 1} out of {NUM_RUNS}.")

        for i, dimensions in enumerate(all_dimensions):
            log_file = open(log_path, "a+")
            log_file.write(f"DIMENSIONS: {dimensions}\n")
            log_file.close()

            num_constraints = None

            if problem_type == "simplexn":
                problem = simplexn(dimensions)
                num_constraints = dimensions * (dimensions - 1) + 1
            elif problem_type == "cuben":
                problem = cuben(dimensions)
                num_constraints = 2 * dimensions
            elif problem_type == "pollution":
                problem = pollutionreduction()
                num_constraints = 3
            elif problem_type == "police":
                problem = police()
                # 10 constraints, 3 of which are redundant
                num_constraints = 7

            # We need at most max/2 positive and max/2 negative samples
            max_samples = max(SAMPLE_SIZES)
            all_true_samples, all_false_samples = get_samples(problem, max_samples // 2, max_samples // 2)
            # all_true_intervals = create_intervals(problem.domain, all_true_samples, 0.1)

            if verbose:
                print(f"Created {max_samples} samples.")

            for j, sample_size in enumerate(SAMPLE_SIZES):

                true_samples = all_true_samples[:sample_size // 2]
                false_samples = all_false_samples[:sample_size // 2]
                # true_intervals = all_true_intervals[:sample_size // 2]

                objective_f = create_objective_function(problem)

                # Calculating the true optimal objective value using the underlying problem
                true_smtlib_problem = convert_to_smtlib(problem.theory, problem.domain, objective_f, optimisation_goal)
                true_f = optimise_model(true_smtlib_problem)

                if verbose:
                    print(f"The true objective value is {true_f}.")

                # Creating a model with IncalP using examples
                tic = time.perf_counter()
                model = run_incalp(problem.domain, true_samples + false_samples, num_constraints)
                toc = time.perf_counter()
                # Using IncalP's model to calculate optimal objective value
                smtlib_problem = convert_to_smtlib(model, problem.domain, objective_f, optimisation_goal)
                tuc = time.perf_counter()
                incalp_estimated_f = optimise_model(smtlib_problem)
                tac = time.perf_counter()
                # Time IncalP took to create the model and find optimal objective value
                incalp_runtimes[i, j, run] = (toc - tic) + (tac - tuc)

                if verbose:
                    print(f"IncalP took {toc - tic:0.2f} + {tac - tuc:0.2f} seconds for {sample_size} samples.")
                    print(f"IncalP-estimated objective value: {incalp_estimated_f}.")

                # Using implicit learning with PAC to find the optimal objective value
                tec = time.perf_counter()
                pac_estimated_f = run_pac(true_samples, objective_f, optimisation_goal)
                tyc = time.perf_counter()
                pac_runtimes[i, j, run] = tyc - tec

                if verbose:
                    print(f"PAC took {tyc - tec:0.2f} seconds for {sample_size} samples.")
                    print(f"PAC-estimated objective value: {pac_estimated_f}.\n")

                log_file = open(log_path, "a")
                log_file.write(f"SAMPLES: {sample_size}\n"
                               f"True f: {true_f}\n"
                               f"IncalP f: {incalp_estimated_f}\n"
                               f"PAC f:f: {pac_estimated_f}\n"
                               f"IncalP time: {(toc - tic) + (tac - tuc)} seconds\n"
                               f"PAC time: {tyc - tec}\n")
                log_file.close()

    # Calculating mean and standard deviation of all the running times
    mean_incalp_runtimes = np.mean(incalp_runtimes, axis=2)
    mean_pac_runtimes = np.mean(pac_runtimes, axis=2)
    std_incalp_runtimes = np.std(incalp_runtimes, axis=2)
    std_pac_runtimes = np.std(pac_runtimes, axis=2)

    # Plotting running times
    if problem_type in ("simplexn", "cuben"):
        fig, axs = plt.subplots(1, 3, sharey='all', constrained_layout=True, figsize=(12, 3))
        for j, ax in enumerate(axs):
            create_plot(ax, SAMPLE_SIZES, "Sample size", mean_incalp_runtimes[j, :], mean_pac_runtimes[j, :],
                        std_incalp_runtimes[j, :], std_pac_runtimes[j, :], "IncalP(SMT)", "PAC")
            ax.title.set_text(f"n: {j + 2}")
        plt.setp(axs[0], ylabel='Time (s)')
        axs[0].legend()
        fig.suptitle(problem.name)
    else:
        fig, ax = plt.subplots(figsize=(5, 4))
        create_plot(ax, SAMPLE_SIZES, "Sample size", mean_incalp_runtimes[0, :], mean_pac_runtimes[0, :],
                    std_incalp_runtimes[0, :], std_pac_runtimes[0, :], "IncalP(SMT)", "PAC")
        ax.set_ylabel('Time (s)')
        ax.legend()

    plot_file = f"output/{random_string}_{problem.name}.pdf"
    plt.savefig(plot_file)

    print(f"Finished! The plot can be found at {plot_file}.")


if __name__ == "__main__":
    main()
