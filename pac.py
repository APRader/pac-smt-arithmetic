from z3 import *
import math
import pandas as pd
import numpy as np


class PAC:
    def __init__(self, z3_vars, knowledge_base):
        self.z3_vars = z3_vars
        self.knowledge_base = knowledge_base

    def decide_pac(self, examples, query, validity):
        state = 'Accept'
        if 0 <= validity <= 1:
            epsilon = 1 - validity
        else:
            raise ValueError('Validity must be between 0 and 1.')
        b = math.floor(epsilon * len(examples))
        failed = 0
        s = Solver()
        s.add(And(self.knowledge_base, Not(query)))
        s.push()
        for example in examples:
            s.add(example)
            # if sat, then the entailment is rejected
            if s.check() == sat:
                failed += 1
                if failed > b:
                    state = 'Reject'
            s.pop()
            s.push()
        return state, (1-failed/len(examples))


# returns number of samples needed to guarantee required confidence and gamma
def sample_size(confidence, gamma):
    if 0 < confidence <= 1:
        delta = 1 - confidence
    else:
        raise ValueError('Confidence must be >0 and <=1')
    if 0 < gamma <= 1:
        return math.ceil(1 / (2 * gamma * gamma) * math.log(1 / delta))
    else:
        raise ValueError('Gamma must be >0 and <=1')


def create_examples(dataset, compression=1):
    """
    Turn dataset into examples in the form of inequalities that can be processed by Z3.
    :param dataset: The dataset to convert, in form of a pandas dataframe
    :param compression: How many rows of the original dataset will be turned into one example.
                        If more than 1, the example will summarize the rows in the following way:
                        The variable will be an interval between the min and max value of the rows compressed
    :return: a dataframe of examples in the form of pairs of numbers (min, max)
    """
    df = pd.DataFrame(columns=dataset.columns)
    grouped_set = dataset.groupby(dataset.index // compression)
    mins = grouped_set.min()
    maxs = grouped_set.max()
    return mins, maxs


def is_in_range(min_examples, max_examples, min_observation, max_observation):
    min_rows = (min_observation >= min_examples) & (min_observation <= max_examples)
    max_rows = (max_observation <= max_examples) & (max_observation >= min_examples)
    all_rows = min_rows | max_rows
    #print(f"Example minimum: {min_examples.iloc[0]}")
    #print(f"KB minimum: {min_observation}")
    #print(f"is KB min > example min? {min_rows.iloc[0]}")
    #print(f"Example maximum: {max_examples.iloc[0]}")
    #print(f"KB maximum: {max_observation}")
    #print(f"is KB max < example min? {max_rows.iloc[0]}")
    #print(all_rows)
    true_rows = all_rows.all(axis=1)
    true_indices = [idx for (idx, entry) in true_rows.iteritems() if entry]
    return true_indices


def mask_examples(examples, probability):
    masked_examples = examples.mask(np.random.random(examples.shape) < probability)
    return masked_examples
