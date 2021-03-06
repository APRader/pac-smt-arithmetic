import time
import motor_util
import pac
import matplotlib.pyplot as plt

KB_COMPRESSION = 1  # compression factor for knowledge base, represents current observations
EXAMPLE_COMPRESSION = 500

z3_vars = motor_util.set_up_variables()

tic = time.perf_counter()

# Load dataset
df_train, df_test = motor_util.load_dataset()
toc = time.perf_counter()
print(f"Loaded dataset in {toc - tic:0.1f} seconds.")

kb = motor_util.compress_dataset(df_test, KB_COMPRESSION)
examples = motor_util.compress_dataset(df_train, EXAMPLE_COMPRESSION)
tuc = time.perf_counter()
print(f"Compressed dataset in {tuc - toc:0.1f} seconds.")

confidence = 0.9
gamma = 0.05
validity = 0.75
number_of_examples = pac.sample_size(confidence, gamma)
print(f"{number_of_examples} examples needed for a confidence of {confidence} and gamma of {gamma}.")
query = z3_vars.get("pm") - z3_vars.get("ambient") > 0

print(f"{len(kb)//2} observations and {len(examples)//2} examples.")

tuc = time.perf_counter()
matched_examples = pac.is_in_range(
    examples.drop(['profile_id'], axis=1), kb.drop(['profile_id'], axis=1))

plt.hist(matched_examples)
plt.title(f"Example compression of {EXAMPLE_COMPRESSION} "
          f"and observation compression of {KB_COMPRESSION}")
plt.xlabel("Number of matched examples")
plt.ylabel("Count")
plt.savefig(f"Ob{KB_COMPRESSION}Ex{EXAMPLE_COMPRESSION}.png")

tac = time.perf_counter()

print(f"Total running time: {tac - tic:0.1f} seconds.")

