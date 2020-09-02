import math
from pysmt.shortcuts import And, Not

class PACLearner:
    def __init__(self, knowledge_base=True):
        self.knowledge_base = knowledge_base

    def decide_pac(self, examples, query, validity):
        """
        PAC decision procedure.
        :param examples: Examples in the form of a list of SMT formulas.
        :param query: SMT formula whose entailment will be decided given examples and background knowledge.
        :param validity: A number from 0 to 1 that represents the validity. Higher number means higher validity.
        :return: True or False, whether the proportion of examples that entailed the query is higher than validity.
        """
        state = 'Accept'
        if 0 <= validity <= 1:
            epsilon = 1 - validity
        else:
            raise ValueError('Validity must be between 0 and 1.')
        b = math.floor(epsilon * len(examples))
        failed = 0
        formula = And(self.knowledge_base, Not(query))
        for example in examples:
            s.add(example)
            # if sat, then the entailment is rejected
            #print(f"Solver: {s}")
            if s.check() == sat:
                #print(f"Model: {s.model()}")
                failed += 1
                if failed > b:
                    state = 'Reject'
            s.pop()
            s.push()
        return state, (1 - failed / len(examples))