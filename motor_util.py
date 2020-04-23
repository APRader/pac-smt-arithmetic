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


def compress_dataset(dataset, compression):
    """
    Compress the dataset while respecting profile_id boundaries.
    :param dataset: The dataset to convert, in form of a pandas dataframe.
    :param compression: How many rows of the original dataset will be turned into one row.
    :return: Two dataframes min and max, which contain the minimum and maximum values for each interval per row.
    """
    min_data = pd.DataFrame()
    max_data = pd.DataFrame()
    for profile_id in dataset.profile_id.unique():
        data_slice = dataset[dataset['profile_id'] == profile_id]
        min_slice, max_slice = pac.create_examples(data_slice.drop(['profile_id'], axis=1), compression)
        # add profile_id column back
        min_slice['profile_id'] = profile_id
        max_slice['profile_id'] = profile_id
        min_data = min_data.append(min_slice)
        max_data = max_data.append(max_slice)
    return min_data, max_data


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


