from z3 import *
import math
import numpy as np


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
    :return: Two dataframes mins and maxs, which contain the minimum and maximum values for each interval per row.
    """
    grouped_set = dataset.groupby(dataset.index // compression)
    mins = grouped_set.min()
    maxs = grouped_set.max()
    return mins, maxs


def is_in_range(min_examples, max_examples, min_observations, max_observations):
    """
    Check how many examples each observation is in range of.
    :param min_examples: A dataframe containing the minimum bounds of each variable. Each row represents one example.
    :param max_examples: A dataframe containing the maximum bounds of each variable. Each row represents one example.
    :param min_observations: A dataframe with the minimum bounds of each variable. Each row represents one observation.
    :param max_observations: A dataframe with the maximum bound sof each variable. Each row represents one observation.
    :return: A list containing the number of examples each observation is in range of.
    """
    # The interval for the observation has to be within the example interval.
    min_indices = min_observations.apply(find_min_indices, args=(min_examples,), axis=1)
    max_indices = max_observations.apply(find_max_indices, args=(max_examples,), axis=1)
    true_indices = [len(min_indices[i].intersection(max_indices[i])) for i in range(len(min_indices))]
    return true_indices


def find_min_indices(min_observation, min_examples):
    """
    Find the indices for which the observation has minimum values above the example minimum values.
    :param min_observation: A series with the minimum bound of each variable. Contains one observation.
    :param min_examples: A dataframe containing the minimum bounds of each variable. Each row represents one example.
    :return: The indices of the examples where the observation had higher values for each variable.
    """
    min_rows = (min_observation >= min_examples).all(axis=1)
    min_indices = min_rows.index[min_rows]
    return min_indices


def find_max_indices(max_observation, max_examples):
    """
    Find the indices for which the observation has a maximum value below the example maximum values.
    :param max_observation: A series with the maximum bound of each variable. Contains one observation.
    :param max_examples: A dataframe containing the maximum bounds of each variable. Each row represents one example.
    :return: The indices of the examples where the observation had lower values for each variable.
    """
    max_rows = (max_observation <= max_examples).all(axis=1)
    max_indices = max_rows.index[max_rows]
    return max_indices


def random_masking(examples, probability):
    """
    Randomly masks values by replacing them with NaN.
    :param examples: A dataframe containing values.
    :param probability: A number from 0 to 1 representing the probability of each entry getting masked.
    :return: A dataframe with some values replaced by NaN.
    """
    masked_examples = examples.mask(np.random.random(examples.shape) < probability)
    return masked_examples
