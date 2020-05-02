import time
import motor_util
import pac
import matplotlib.pyplot as plt

KB_COMPRESSION = 10  # compression factor for knowledge base, represents current observations
EXAMPLE_COMPRESSION = 10000

z3_vars = motor_util.set_up_variables()

tic = time.perf_counter()

# Load dataset
df_train, df_test = motor_util.load_dataset()
toc = time.perf_counter()
print(f"Loaded dataset in {toc - tic:0.1f} seconds.")

min_kb, max_kb = motor_util.compress_dataset(df_test, KB_COMPRESSION)
min_examples, max_examples = motor_util.compress_dataset(df_train, EXAMPLE_COMPRESSION)
tuc = time.perf_counter()
print(f"Compressed dataset in {tuc - toc:0.1f} seconds.")

confidence = 0.9
gamma = 0.05
validity = 0.75
number_of_examples = pac.sample_size(confidence, gamma)
print(f"{number_of_examples} examples needed for a confidence of {confidence} and gamma of {gamma}.")
query = z3_vars.get("pm") - z3_vars.get("ambient") > 0

tuc = time.perf_counter()
matched_examples = []
for i in range(len(min_kb)):
    indices = pac.is_in_range(
        min_examples.drop(['profile_id'], axis=1), max_examples.drop(['profile_id'], axis=1),
        min_kb.drop(['profile_id'], axis=1).iloc[i], max_kb.drop(['profile_id'], axis=1).iloc[i])
    matched_examples.append(len(indices))
    if i % 1000 == 0:
        tac = time.perf_counter()
        print(f"First {i} observations processed in {tac - tuc:0.1f} seconds.")

plt.hist(matched_examples)
plt.title(f"Matched examples for example compression of {EXAMPLE_COMPRESSION} "
          f"and observation compression of {KB_COMPRESSION}")
plt.xlabel("Number of matched examples")
plt.ylabel("Count")
plt.show()
