from z3 import Solver, And, Not, sat
import math
from pac import interval as interval
from fractions import Fraction


class PACLearner:
    def __init__(self, knowledge_base=True, domain=None):
        # Knowledge base can be Z3 formula
        self.knowledge_base = knowledge_base
        # Domain can be Z3 formula, Interval object or a list of those
        if domain and type(domain) is not list:
            self.domain = [domain]
        else:
            self.domain = domain

    def decide_pac(self, examples, query, validity=1.0, return_actual_validity=False):
        """
        PAC decision procedure.
        :param examples: Examples in the form of a list of Z3 formulas or list of list of Interval objects.
        :param query: Z3 formula whose entailment will be decided given examples and background knowledge.
        :param validity: A number from 0 to 1 that represents the validity. Higher number means higher validity.
        :param return_actual_validity: Whether to return the actual validity in addition to the decision.
        :return: True if it accepts, False otherwise.
        """
        state = True

        if 0 <= validity <= 1:
            epsilon = 1 - validity
        else:
            raise ValueError('Validity must be between 0 and 1.')

        b = math.floor(epsilon * len(examples))
        failed = 0
        s = Solver()
        s.add(self.knowledge_base, Not(query))
        if self.domain:
            if isinstance(self.domain[0], interval.Interval):
                s.add(And([dom.create_formula() for dom in self.domain]))
            else:
                s.add(And(self.domain))
        s.push()
        for example in examples:
            if type(example) is list:
                # Assumed to be list of Interval objects
                s.add(And([inter.create_formula() for inter in example]))
            else:
                s.add(example)
            # If sat, then the entailment is rejected
            if s.check() == sat:
                failed += 1
                if failed > b:
                    if not return_actual_validity:
                        # We can stop now, as the query cannot be valid anymore
                        return False
                    else:
                        state = False
            s.pop()
            s.push()
        if return_actual_validity:
            return state, (1 - failed / len(examples))
        else:
            return state

    def optimise_pac(self, examples, objective_f, goal="maximise", validity=1.0, accuracy=60):
        """
        Find the optimal objective value given positive examples.
        :param accuracy: Number of iterations of halving the bounded interval.
        :param examples: Positive samples.
        :param objective_f: Z3 formula for the objective function we want to optimise.
        :param goal: Whether the goal is to maximise or minimise the objective function.
        :param validity: The ratio of samples that must be valid against the queries.
        :return: The estimated optimal value of the objective function.
        """
        if goal not in ("maximise", "minimise"):
            raise ValueError("Goal must either be 'maximise' or 'minimise'.")

        if goal == "minimise":
            # Minimisation is the same as maximisation with objective function of opposite sign
            objective_f = -objective_f

        if self.decide_pac(examples, 0 >= objective_f, validity):
            # Optimal objective value is negative or 0
            if not self.decide_pac(examples, -1 >= objective_f, validity):
                lower = -1
                upper = 0
            else:
                # Optimal objective value is less than -1, finding rough bounds using exponential search
                bound = -2
                while self.decide_pac(examples, bound >= objective_f, validity):
                    bound *= 2
                lower = bound
                upper = bound / 2
        else:
            # Optimal objective value is positive
            if self.decide_pac(examples, 1 >= objective_f, validity):
                lower = 0
                upper = 1
            else:
                # Optimal objective value is greater than 1, finding rough bounds using exponential search
                bound = 2
                while not self.decide_pac(examples, bound >= objective_f, validity):
                    bound *= 2
                lower = bound / 2
                upper = bound

        # Converting the bounds into fractions to allow for exact representations
        upper = Fraction(upper)
        lower = Fraction(lower)

        # Finding tight bounds for optimal objective value using binary search
        for i in range(accuracy):
            if self.decide_pac(examples, (lower + upper) / 2 >= objective_f, validity):
                upper = (lower + upper) / 2
            else:
                lower = (lower + upper) / 2

        if goal == "minimise":
            # Flipping sign back again
            return -lower
        else:
            return lower
