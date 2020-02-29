from z3 import *
import math
import pandas as pd
import numpy as np


def decide_pac(bg_knowledge, examples, query, validity):
    if 0 <= validity <= 1:
        epsilon = 1 - validity
    else:
        raise ValueError('Validity must be between 0 and 1.')
    b = math.floor(epsilon * len(examples))
    failed = 0
    s = Solver()
    s.add(And(bg_knowledge, Not(query)))
    s.push()
    for example in examples:
        s.add(example)
        # if sat, then the entailment is rejected
        if s.check() == sat:
            failed += 1
            if failed > b:
                return 'Reject'
        s.pop()
        s.push()
    return 'Accept'


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

def is_in_range(observation, example):
    #print(observation)
    return None
