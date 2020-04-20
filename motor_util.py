import pandas as pd
from z3 import *
import pac
import time


def load_dataset():
    """
    Loads the Electric Motor Temperature dataset found in https://www.kaggle.com/wkirgsn/electric-motor-temperature.
    Assumes that the file 'pmsm_temperature_data.csv' is in the same folder.
    :return: The train set and test set as pandas dataframes.
    """
    df = pd.read_csv('pmsm_temperature_data.csv')
    train_set = df.loc[~df['profile_id'].isin([65, 72])]
    test_set = df.loc[df['profile_id'].isin([65, 72])]
    return train_set, test_set


def compress_dataset(dataset, train_compression, test_compression):
    """
    Randomly masks values by replacing them with NaN.
    :param examples: A dataframe containing values.
    :param probability: A number from 0 to 1 representing the probability of each entry getting masked.
    :return: A dataframe with some values replaced by NaN.
    """
    tic = time.perf_counter()

    df = pd.read_csv('pmsm_temperature_data.csv')
    # print(len(df.index))

    toc = time.perf_counter()
    print(f"Read the dataset in in {toc - tic:0.1f} seconds.")

    # train_set = df.loc[~df['profile_id'].isin([65, 72])]
    # test_set = df.loc[df['profile_id'].isin([65, 72])]

    train_set = df.loc[df['profile_id'].isin([4])]
    test_set = train_set[0:30000]
    train_set = train_set[30000:]

    min_train_set, max_train_set = pac.create_examples(train_set.drop(['profile_id'], axis=1), train_compression)
    # train_set.to_csv(FILE_PATH + fr"\train_set{train_compression}.tsv", sep="\t")
    tuc = time.perf_counter()
    print(f"Created train set in {tuc - toc:0.1f} seconds.")
    min_test_set, max_test_set = pac.create_examples(test_set.drop(['profile_id'], axis=1), test_compression)
    # test_set.to_csv(FILE_PATH + fr"\test_set{test_compression}.tsv", sep="\t")
    tac = time.perf_counter()
    print(f"Created test set in {tac - tuc:0.1f} seconds.")
    print(f"Converted the whole dataset in {tac - toc:0.1f} seconds")
    return min_train_set, max_train_set, min_test_set, max_test_set


def match_examples(min_observation_feats, max_observation_feats, min_example_feats, max_example_feats):
    matched_examples = []
    tic = time.perf_counter()
    for i in range(1000):  # len(min_observations)):
        min_observation = min_observation_feats.iloc[i]
        max_observation = max_observation_feats.iloc[i]
        indices = pac.is_in_range(min_example_feats, max_example_feats, min_observation, max_observation)
        # print(indices)
        # for index in indices:
        #    print(f"actual temperature between {min_observations.iloc[index]} and {max_observations.iloc[index]}")
        matched_examples.append(len(indices))
        if i % 100 == 0:
            toc = time.perf_counter()
            print(f"First {i} observations processed in {toc - tic:0.1f} seconds.")
        # print(f"The observation is in range of {len(indices)} examples")
    toc = time.perf_counter()
    # print(f"All {len(min_observations)} observations processed in {toc - tic:0.1f} seconds..")
    return matched_examples


def create_formulas():
    tic = time.perf_counter()

    df = pd.read_csv('pmsm_temperature_data.csv')

    toc = time.perf_counter()
    print(f"Read the dataset in in {toc - tic:0.1f} seconds.")

    data = df.loc[df['profile_id'] == 4]
    compression = len(data)
    min_val, max_val = pac.create_examples(data, compression)
    knowledge_base = And(ambient >= min_val.at[0, "ambient"], ambient <= max_val.at[0, "ambient"],
                         pm >= min_val.at[0, "pm"], pm <= max_val.at[0, "pm"])

    tuc = time.perf_counter()
    print(f"Created knowledge base in {tuc - toc:0.1f} seconds.")

    compression = 10
    min_vals, max_vals = pac.create_examples(data, compression)
    min_examples = pac.random_masking(min_vals, 0.0)
    max_examples = pac.random_masking(min_vals, 0.0)

    return knowledge_base, min_examples, max_examples

