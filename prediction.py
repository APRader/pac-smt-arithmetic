import time
import motor_util
import pac
import matplotlib.pyplot as plt

KB_COMPRESSION = 5  # compression factor for knowledge base, represents current observations
EXAMPLE_COMPRESSION = 1000

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
matched_examples = pac.is_in_range(
    min_examples.drop(['profile_id'], axis=1), max_examples.drop(['profile_id'], axis=1),
    min_kb.drop(['profile_id'], axis=1), max_kb.drop(['profile_id'], axis=1))

plt.hist(matched_examples)
plt.title(f"Example compression of {EXAMPLE_COMPRESSION} "
          f"and observation compression of {KB_COMPRESSION}")
plt.xlabel("Number of matched examples")
plt.ylabel("Count")
plt.savefig(f"Ob{KB_COMPRESSION}Ex{EXAMPLE_COMPRESSION}.svg")

tac = time.perf_counter()

print(f"Total running time: {tac - tic:0.1f} seconds.")

