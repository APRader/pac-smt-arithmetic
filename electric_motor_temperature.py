import pandas as pd
from z3 import *
import pac
import time
import matplotlib.pyplot as plt

FILE_PATH = r'C:\Users\APRader\PycharmProjects\pac-smt-arithmetic\datasets'
TRAIN_COMPR = 20
TEST_COMPR = 2

def compress_dataset(train_compression, test_compression):
    tic = time.perf_counter()

    df = pd.read_csv('pmsm_temperature_data.csv')
    # print(len(df.index))

    toc = time.perf_counter()
    print(f"Read the dataset in in {toc - tic:0.1f} seconds.")

    train_set = df.loc[~df['profile_id'].isin([65, 72])]
    test_set = df.loc[df['profile_id'].isin([65, 72])]
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


def read_dataset(train_compression, test_compression):
    train_set = pd.read_csv(FILE_PATH + fr"\train_set{train_compression}.tsv", sep="\t")
    test_set = pd.read_csv(FILE_PATH + fr"\test_set{test_compression}.tsv", sep="\t")
    return train_set, test_set


ambient, coolant, u_d, u_q, motor_speed, torque, i_d, i_q, pm, stator_yoke, stator_tooth, stator_winding = \
    Reals('ambient coolant u_d u_q motor_speed torque i_d i_q pm stator_yoke stator_tooth stator_winding')
z3_vars = [ambient, coolant, u_d, u_q, motor_speed, torque, i_d, i_q, pm, stator_yoke, stator_tooth, stator_winding]
set_option(rational_to_decimal=True)

min_examples, max_examples, min_observations, max_observations = compress_dataset(TRAIN_COMPR, TEST_COMPR)
# examples, observations = read_dataset(20, 5)

print(f"{len(min_examples)} examples.")
print(f"{len(min_observations)} data points in the test set.")

target_features = ['pm', 'stator_tooth', 'stator_yoke', 'stator_winding']

confidence = 0.9
gamma = 0.05
number_of_examples = pac.sample_size(confidence, gamma)
print(f"{number_of_examples} examples needed for a confidence of {confidence} and gamma of {gamma}.")
query = pm > 0

min_observation_feats = min_observations.drop(target_features, axis=1)
max_observation_feats = max_observations.drop(target_features, axis=1)
min_example_feats = min_examples.drop(target_features, axis=1)
max_example_feats = max_examples.drop(target_features, axis=1)

matched_examples = []
tic = time.perf_counter()
for i in range(1000):#len(min_observations)):
    min_observation = min_observation_feats.iloc[i]
    max_observation = max_observation_feats.iloc[i]
    indices = pac.is_in_range(min_example_feats, max_example_feats, min_observation, max_observation)
    #print(indices)
    #for index in indices:
    #    print(f"actual temperature between {min_observations.iloc[index]} and {max_observations.iloc[index]}")
    matched_examples.append(len(indices))
    if i % 100 == 0:
        toc = time.perf_counter()
        print(f"First {i} observations processed in {toc - tic:0.1f} seconds.")
    # print(f"The observation is in range of {len(indices)} examples")

toc = time.perf_counter()
print(f"All {len(min_observations)} observations processed in {toc - tic:0.1f} seconds..")

plt.hist(matched_examples)
plt.title(f"Matched examples for example compression of {TRAIN_COMPR} and observation compression of {TEST_COMPR}")
plt.xlabel("Number of matched examples")
plt.ylabel("Count")
plt.show()
'''
for index, row in min_examples.iterrows():
    pac.is_in_range(min_examples, max_examples, row)
    bg_knowledge = test_set[0]
    truth = target_values[0]
    print(pac.decide_pac(bg_knowledge, examples, query, 0.61))
    '''


