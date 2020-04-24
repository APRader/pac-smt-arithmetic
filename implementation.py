import motor_util
import time
import pac
from z3 import *
import pandas as pd
import matplotlib.pyplot as plt

KB_COMPRESSION = 1000000  # compression factor for knowledge base, larger than amount of data points in any profile
EXAMPLE_COMPRESSION = 10
MASKING_PROBABILITIES = [0.1] #[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

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

query = i_d + u_d + i_q - u_q > 0
validity = 0.8
print(f"Validity set to {validity}.")
validities = pd.DataFrame(columns=min_kb.profile_id.unique()[:10], index=MASKING_PROBABILITIES)

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
    for masking in MASKING_PROBABILITIES:
        tuc = time.perf_counter()
        print(f"Masking probability of {masking:0.1f}.")

        mins_masked = pac.random_masking(mins, masking)
        maxs_masked = pac.random_masking(maxs, masking)
        examples = pac_object.create_inequalities(mins_masked, maxs_masked)
        tec = time.perf_counter()
        decision, prop_valid = pac_object.decide_pac(examples, query, validity)
        validities.at[masking, profile_id] = prop_valid
        tac = time.perf_counter()
        print(f"PAC decided to {decision} since {prop_valid:0.3f} of the examples were valid "
              f"after thinking for {tac - tuc:0.1f} seconds.")
        print(f"The decision procedure itself took {tac - tec:0.1f} seconds.")

ax = plt.gca()
validities.plot(kind='line', ax=ax)
plt.xlabel('Masking')
plt.ylabel('Validity')
plt.show()
tac = time.perf_counter()
print(f"The entire process took {tac - tic:0.1f} seconds.")
