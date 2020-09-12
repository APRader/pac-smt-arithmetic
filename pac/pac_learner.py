from z3 import Solver, And, Not, sat
import math


class PACLearner:
    def __init__(self, knowledge_base=True, domain=None):
        self.knowledge_base = knowledge_base
        self.domain = domain

    def decide_pac(self, examples, query, validity=1, return_actual_validity=False):
        """
        PAC decision procedure.
        :param examples: Examples in the form of a list of Z3 formulas.
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
        s.push()
        for example in examples:
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
