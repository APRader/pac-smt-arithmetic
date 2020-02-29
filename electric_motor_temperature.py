import pandas as pd
from z3 import *
import pac
import time


def compress_dataset(train_compression, test_compression):
    tic = time.perf_counter()

    df = pd.read_csv('pmsm_temperature_data.csv')
    #print(len(df.index))

    toc = time.perf_counter()
    print(f"Read the dataset in in {toc - tic:0.1f} seconds.")

    train_set = df.loc[df['profile_id'] != (65 or 72)]
    test_set = df.loc[df['profile_id'] == (65 or 72)]
    train_set = pac.create_examples(train_set, train_compression)
    train_set.to_csv(r'train_set.csv')
    tuc = time.perf_counter()
    print(f"Created testset in {tuc - toc:0.1f} seconds.")
    test_set = pac.create_examples(test_set, test_compression)
    test_set.to_csv(r'test_set.csv')
    tac = time.perf_counter()
    print(f"Created train set in {tac - tuc:0.1f} seconds.")
    print(f"Converted the whole dataset in {tac - toc:0.1f} seconds")
    return None


def read_dataset():
    examples = []
    with open('examples.txt', 'r') as filehandle:
        examples = [example.rstrip() for example in filehandle.readlines()]
    return examples


ambient, coolant, u_d, u_q, motor_speed, torque, i_d, i_q, pm, stator_yoke, stator_tooth, stator_winding = \
    Reals('ambient coolant u_d u_q motor_speed torque i_d i_q pm stator_yoke stator_tooth stator_winding')
z3_vars = [ambient, coolant, u_d, u_q, motor_speed, torque, i_d, i_q, pm, stator_yoke, stator_tooth, stator_winding]
set_option(rational_to_decimal=True)

compress_dataset(20, 5)

'''

print(f"{len(examples)} examples.")
print(f"{len(test_set)} data points in the test set.")
print(f"{len(target_values)} target values.")


# Kaggle task says to drop torque, since it is not reliably measurable.
# But PAC can deal with imperfect data, so we will use it anyway.
# target_features = ['pm', 'stator_tooth', 'stator_yoke', 'stator_winding']

# print(profile_65.head(100))
confidence = 0.9
number_of_examples = pac.sample_size(confidence, 0.01)
print(f"{number_of_examples} examples needed for a confidence of {confidence}.")
query = pm > 0

for example in examples:
    print(example)
    bg_knowledge = test_set[0]
    truth = target_values[0]
    print(pac.decide_pac(bg_knowledge, examples, query, 0.61))
'''