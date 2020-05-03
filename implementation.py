import motor_util
import time
import pac
import pandas as pd
import matplotlib.pyplot as plt

KB_COMPRESSION = 1000000  # compression factor for knowledge base, larger than amount of data points in any profile
EXAMPLE_COMPRESSION = 10
MASKING_PROBABILITIES = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

z3_vars = motor_util.set_up_variables()

tic = time.perf_counter()

# Load dataset
df_train, _ = motor_util.load_dataset()
toc = time.perf_counter()
print(f"Loaded dataset in {toc - tic:0.1f} seconds.")

# For knowledge base, each profile will create one min and one max
kb = motor_util.compress_dataset(df_train, KB_COMPRESSION)
examples = motor_util.compress_dataset(df_train, EXAMPLE_COMPRESSION)
tuc = time.perf_counter()
print(f"Compressed dataset in {tuc - toc:0.1f} seconds.")

query = z3_vars.get("i_d") * z3_vars.get("u_d") + z3_vars.get("i_q") * z3_vars.get("u_q") > 0
validity = 0.8
print(f"Validity set to {validity}.")
validities = pd.DataFrame(columns=kb.profile_id.unique()[:10], index=MASKING_PROBABILITIES)

# Do PAC decision procedure for each profile separately
for profile_id in kb.profile_id.unique()[:10]:
    pac_object = pac.PAC(z3_vars)
    current_kb = kb[kb['profile_id'] == profile_id].drop(['profile_id'], axis=1)
    knowledge_base = pac_object.create_inequalities(current_kb)
    pac_object.knowledge_base = knowledge_base[0]

    current_examples = examples[examples['profile_id'] == profile_id].drop(['profile_id'], axis=1)
    print(f"Profile {profile_id} contains {len(current_examples)//2} examples.")
    for masking in MASKING_PROBABILITIES:
        tuc = time.perf_counter()
        print(f"Masking probability of {masking:0.1f}.")

        masked_examples = pac.random_masking(current_examples, masking)
        examples = pac_object.create_inequalities(masked_examples)
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
