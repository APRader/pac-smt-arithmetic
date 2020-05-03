import pandas as pd
import pac
from z3 import *


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
    :return: A dataframe which contains the minimum and maximum values for each interval in an index hierarchy.
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
    min_data.reset_index(inplace=True, drop=True)
    max_data.reset_index(inplace=True, drop=True)
    # add column specifying whether row contains minimum or maximum values
    min_data['bound'] = 'min'
    max_data['bound'] = 'max'
    # create dataframe that contains min and max dataframes with hierarchical index based on bound
    data = pd.concat([min_data, max_data])
    data.set_index(['bound', data.index], inplace=True)
    return data


def set_up_variables():
    """
    Create all Z3 variables from the motor dataset.
    :return: A dictionary of all the variables, where the key is the column name of each variable.
    """
    ambient, coolant, u_d, u_q, motor_speed, torque, i_d, i_q, pm, stator_yoke, stator_tooth, stator_winding = \
        Reals('ambient coolant u_d u_q motor_speed torque i_d i_q pm stator_yoke stator_tooth stator_winding')
    z3_vars = {'ambient': ambient, 'coolant': coolant, 'u_d': u_d, 'u_q': u_q, 'motor_speed': motor_speed,
               'torque': torque, 'i_d': i_d, 'i_q': i_q, 'pm': pm, 'stator_yoke': stator_yoke,
               'stator_tooth': stator_tooth, 'stator_winding': stator_winding}
    set_option(rational_to_decimal=True)
    return z3_vars
