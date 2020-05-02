import motor_util
import pac
import time

tic = time.perf_counter()

# Load dataset
df_train, df_test = motor_util.load_dataset()
toc = time.perf_counter()
print(f"Loaded dataset in {toc - tic:0.1f} seconds.")

min_kb, max_kb = motor_util.compress_dataset(df_test, 100)
min_examples, max_examples = motor_util.compress_dataset(df_train, 5000)
tuc = time.perf_counter()
print(f"Compressed dataset in {tuc - toc:0.1f} seconds.")

tuc = time.perf_counter()

indices = pac.is_in_range(
    min_examples.drop(['profile_id'], axis=1), max_examples.drop(['profile_id'], axis=1),
    min_kb.drop(['profile_id'], axis=1), max_kb.drop(['profile_id'], axis=1))
