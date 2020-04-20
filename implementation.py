import motor_util
import time
import pac
from z3 import *

KB_COMPRESSION = 1000000  # compression factor for knowledge base, larger than amount of data points in any profile
EXAMPLE_COMPRESSION = 10

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
# Do PAC decision procedure for each profile separately
for profile_id in min_kb.profile_id.unique():
    tuc = time.perf_counter()
    mins = min_kb[min_kb['profile_id'] == profile_id].drop(['profile_id'], axis=1)
    maxs = max_kb[max_kb['profile_id'] == profile_id].drop(['profile_id'], axis=1)
    knowledge_base = And([z3_vars.get(col) >= mins.at[0, col] for col in mins.columns] +
                         [z3_vars.get(col) <= maxs.at[0, col] for col in maxs.columns])
    pac_object = pac.PAC(z3_vars, knowledge_base)
    mins = min_examples[min_examples['profile_id'] == profile_id].drop(['profile_id'], axis=1)
    maxs = max_examples[max_examples['profile_id'] == profile_id].drop(['profile_id'], axis=1)
    print(f"Profile {profile_id} contains {len(mins)} examples")
    examples = [And([z3_vars.get(col) >= mins.at[row, col] for col in mins.columns] +
                    [z3_vars.get(col) <= maxs.at[row, col] for col in maxs.columns]) for row in mins.index]
    decision, prop_valid = pac_object.decide_pac(examples, query, validity)
    tac = time.perf_counter()
    print(f"PAC decided to {decision} since {prop_valid} of the examples were valid "
          f"after thinking for {tac - tuc:0.1f} seconds.")

'''


pac_object = pac.PAC(z3_vars, knowledge_base)

print(f"{len(min_examples)} examples.")

confidence = 0.9
gamma = 0.05
validity = 0.75
number_of_examples = pac.sample_size(confidence, gamma)
print(f"{number_of_examples} examples needed for a confidence of {confidence} and gamma of {gamma}.")
query = pm - ambient > 0
examples = []

for index in range(number_of_examples):
    min_ambient = min_examples.at[index, "ambient"]
    max_ambient = max_examples.at[index, "ambient"]
    min_pm = min_examples.at[index, "pm"]
    max_pm = max_examples.at[index, "pm"]
    inequalities = []
    if not (math.isnan(min_ambient)): inequalities.append(ambient >= min_ambient)
    if not (math.isnan(max_ambient)): inequalities.append(ambient <= max_ambient)
    if not (math.isnan(min_pm)): inequalities.append(pm >= min_pm)
    if not (math.isnan(max_pm)): inequalities.append(pm <= max_pm)
    examples.append(And(inequalities))

state, valid_ratio = pac_object.decide_pac(examples, query, validity)
print(f"PAC has spoken: {state}, because {valid_ratio} examples were valid.")


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
