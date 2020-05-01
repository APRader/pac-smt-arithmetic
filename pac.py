from z3 import *
import math
import numpy as np
import time


class PAC:
    def __init__(self, z3_vars, knowledge_base=None):
        if knowledge_base is None:
            knowledge_base = []
        self.z3_vars = z3_vars
        self.knowledge_base = knowledge_base

    def decide_pac(self, examples, query, validity):
        """
        PAC decision procedure.
        :param examples: Examples in the form of a list of Z3 formulas.
        :param query: Z3 formula whose entailment will be decided given examples and background knowledge.
        :param validity: A number from 0 to 1, that represents the validity. Higher number means higher validity.
        :return: The state ("Accept" or "Reject") and the proportion of examples that entailed the query.
        """
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
        return state, (1 - failed / len(examples))

    def create_inequalities(self, min_data, max_data):
        """
        Turn minimum and maximum data into Z3 inequalities.
        :param min_data: Dataframe where each row represents the minimum values of an instance.
        :param max_data: Dataframe where each row represents the maximum values of an instance.
        :return: A list of Z3 formulas that are conjunctions of inequalities.
        """
        return [And(
            [self.z3_vars.get(col) >= min_data.at[row, col] for col in min_data.columns
             if not math.isnan(min_data.at[row, col])] +
            [self.z3_vars.get(col) <= max_data.at[row, col] for col in max_data.columns
             if not math.isnan(max_data.at[row, col])])
            for row in min_data.index]


def sample_size(confidence, gamma):
    """
    Calculates number of examples needed to guarantee required confidence and deviation.
    :param confidence: Number from 0 to 1 which represents the confidence. Higher number means higher confidence.
    :param gamma: Number from 0 to 1 which represents the amount of deviation allowed from estimate.
    :return: Number of examples needed.
    """
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
    Turn dataset into examples in the form of minimum and maximum values.
    :param dataset: The dataset to convert, in form of a pandas dataframe.
    :param compression: How many rows of the original dataset will be turned into one example.
    :return: Two dataframes min and max, which contain the minimum and maximum values for each interval per row.
    """
    grouped_set = dataset.groupby(dataset.index // compression)
    mins = grouped_set.min()
    maxs = grouped_set.max()
    return mins, maxs


def is_in_range(min_examples, max_examples, min_observation, max_observation):
    """
    Check which examples an observation is in range of for each variable.
    :param min_examples: A dataframe containing the minimum bounds of each variable. Each row represents one example.
    :param max_examples: A dataframe containing the maximum bounds of each variable. Each row represents one example.
    :param min_observation: A series containing the minimum bound of each variable. Contains one observation.
    :param max_observation: A series containing the maximum bound of each variable. Contains one observation.
    :return: The row indices for the examples that matched the observation.
    """
    # The interval for the observation has to be within the example interval.
    true_rows = ((min_observation >= min_examples) & (max_observation <= max_examples)).all(axis=1)
    true_indices = true_rows.index[true_rows]
    return true_indices


def random_masking(examples, probability):
    """
    Randomly masks values by replacing them with NaN.
    :param examples: A dataframe containing values.
    :param probability: A number from 0 to 1 representing the probability of each entry getting masked.
    :return: A dataframe with some values replaced by NaN.
    """
    masked_examples = examples.mask(np.random.random(examples.shape) < probability)
    return masked_examples
