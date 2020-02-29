import pandas as pd
from z3 import *
import pac
import time

FILE_PATH = r'C:\Users\APRader\PycharmProjects\pac-smt-arithmetic\datasets'


def compress_dataset(train_compression, test_compression):
    tic = time.perf_counter()

    df = pd.read_csv('pmsm_temperature_data.csv')
    # print(len(df.index))

    toc = time.perf_counter()
    print(f"Read the dataset in in {toc - tic:0.1f} seconds.")

    train_set = df.loc[~df['profile_id'].isin([65, 72])]
    test_set = df.loc[df['profile_id'].isin([65, 72])]
    train_set = pac.create_examples(train_set, train_compression)
    train_set.to_csv(FILE_PATH + fr"\train_set{train_compression}.tsv", sep="\t", index=False)
    tuc = time.perf_counter()
    print(f"Created train set in {tuc - toc:0.1f} seconds.")
    test_set = pac.create_examples(test_set, test_compression)
    test_set.to_csv(FILE_PATH + fr"\test_set{test_compression}.tsv", sep="\t", index=False)
    tac = time.perf_counter()
    print(f"Created test set in {tac - tuc:0.1f} seconds.")
    print(f"Converted the whole dataset in {tac - toc:0.1f} seconds")
    return None


def read_dataset(train_compression, test_compression):
    train_set = pd.read_csv(FILE_PATH + fr"\train_set{train_compression}.tsv", sep="\t")
    test_set = pd.read_csv(FILE_PATH + fr"\test_set{test_compression}.tsv", sep="\t")
    return train_set, test_set


ambient, coolant, u_d, u_q, motor_speed, torque, i_d, i_q, pm, stator_yoke, stator_tooth, stator_winding = \
    Reals('ambient coolant u_d u_q motor_speed torque i_d i_q pm stator_yoke stator_tooth stator_winding')
z3_vars = [ambient, coolant, u_d, u_q, motor_speed, torque, i_d, i_q, pm, stator_yoke, stator_tooth, stator_winding]
set_option(rational_to_decimal=True)

#compress_dataset(20, 5)
examples, observations = read_dataset(20, 5)

print(f"{len(examples)} examples.")
print(f"{len(observations)} data points in the test set.")

# target_features = ['pm', 'stator_tooth', 'stator_yoke', 'stator_winding']

confidence = 0.9
gamma = 0.05
number_of_examples = pac.sample_size(confidence, gamma)
print(f"{number_of_examples} examples needed for a confidence of {confidence} and gamma of {gamma}.")
query = pm > 0

'''
for example in examples:
    print(example)
    bg_knowledge = test_set[0]
    truth = target_values[0]
    print(pac.decide_pac(bg_knowledge, examples, query, 0.61))

'''