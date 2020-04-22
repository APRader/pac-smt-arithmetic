import motor_util
import time
import pac
from z3 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

KB_COMPRESSION = 1000000  # compression factor for knowledge base, larger than amount of data points in any profile
EXAMPLE_COMPRESSION = 100

ambient, coolant, u_d, u_q, motor_speed, torque, i_d, i_q, pm, stator_yoke, stator_tooth, stator_winding = \
    Reals('ambient coolant u_d u_q motor_speed torque i_d i_q pm stator_yoke stator_tooth stator_winding')
z3_vars = {'ambient': ambient, 'coolant': coolant, 'u_d': u_d, 'u_q': u_q, 'motor_speed': motor_speed, 'torque': torque,
           'i_d': i_d, 'i_q': i_q, 'pm': pm, 'stator_yoke': stator_yoke, 'stator_tooth': stator_tooth,
           'stator_winding': stator_winding}
set_option(rational_to_decimal=True)

tic = time.perf_counter()

# Load dataset
df_train, df_test = motor_util.load_dataset()
toc = time.perf_counter()
print(f"Loaded dataset in {toc - tic:0.1f} seconds.")

# For knowledge base, each profile will create one min and one max
min_kb, max_kb = motor_util.compress_dataset(df_train, KB_COMPRESSION)
min_examples, max_examples = motor_util.compress_dataset(df_train, EXAMPLE_COMPRESSION)
tuc = time.perf_counter()
print(f"Compressed dataset in {tuc - toc:0.1f} seconds.")

query = ambient - pm > 0
validity = 0.8
print(f"Validity set to {validity}.")
validities = pd.DataFrame(columns=min_kb.profile_id.unique()[:10], index=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

# Do PAC decision procedure for each profile separately
for profile_id in min_kb.profile_id.unique()[:10]:
    pac_object = pac.PAC(z3_vars)
    mins = min_kb[min_kb['profile_id'] == profile_id].drop(['profile_id'], axis=1)
    maxs = max_kb[max_kb['profile_id'] == profile_id].drop(['profile_id'], axis=1)
    knowledge_base = pac_object.create_inequalities(mins, maxs)
    pac_object.knowledge_base = knowledge_base[0]

    mins = min_examples[min_examples['profile_id'] == profile_id].drop(['profile_id'], axis=1)
    maxs = max_examples[max_examples['profile_id'] == profile_id].drop(['profile_id'], axis=1)
    print(f"Profile {profile_id} contains {len(mins)} examples.")
    for masking in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        tuc = time.perf_counter()
        print(f"Masking probability of {masking:0.1f}.")

        mins_masked = pac.random_masking(mins, masking)
        maxs_masked = pac.random_masking(maxs, masking)
        examples = pac_object.create_inequalities(mins_masked, maxs_masked)

        decision, prop_valid = pac_object.decide_pac(examples, query, validity)
        validities.at[masking,profile_id] = prop_valid
        tac = time.perf_counter()
        print(f"PAC decided to {decision} since {prop_valid:0.3f} of the examples were valid "
              f"after thinking for {tac - tuc:0.1f} seconds.")

ax = plt.gca()

validities.plot(kind='line',ax=ax)
plt.xlabel('Masking')
plt.ylabel('Validity')
plt.show()
tac = time.perf_counter()
print(f"The entire process took {tac - tic:0.1f} seconds.")


'''

confidence = 0.9
gamma = 0.05
validity = 0.75
number_of_examples = pac.sample_size(confidence, gamma)
print(f"{number_of_examples} examples needed for a confidence of {confidence} and gamma of {gamma}.")
query = pm - ambient > 0
examples = []


min_examples, max_examples, min_observations, max_observations = compress_dataset(TRAIN_COMPR, TEST_COMPR)
# examples, observations = read_dataset(20, 5)

print(f"{len(min_observations)} data points in the test set.")

target_features = ['pm', 'stator_tooth', 'stator_yoke', 'stator_winding']


min_observation_feats = min_observations.drop(target_features, axis=1)
max_observation_feats = max_observations.drop(target_features, axis=1)
min_example_feats = min_examples.drop(target_features, axis=1)
max_example_feats = max_examples.drop(target_features, axis=1)


matched_examples = match_examples(min_observation_feats, max_observation_feats, min_example_feats, max_example_feats)

plt.hist(matched_examples)
plt.title(f"Matched examples for example compression of {TRAIN_COMPR} and observation compression of {TEST_COMPR}")
plt.xlabel("Number of matched examples")
plt.ylabel("Count")
plt.show()
'''
