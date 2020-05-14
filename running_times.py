import motor_util
import time
import pac
import matplotlib.pyplot as plt

KB_COMPRESSION = 1000000  # compression factor for knowledge base, larger than amount of data points in any profile
EXAMPLE_COMPRESSION = 10
VALIDITY = 0.5
NO_OF_QUERIES = 1
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

pac_object = pac.PAC(z3_vars)
examples = pac_object.create_inequalities(examples.drop(['profile_id'], axis=1))
tac = time.perf_counter()
print(f"Created {len(examples)} inequalities in {tac - tuc:0.1f} seconds.")

running_times = []
for i in range(1, 90001, 5000):
    queries = pac_object.generate_linear_queries(NO_OF_QUERIES, 10)
    current_times = []
    for query in queries:
        tac = time.perf_counter()
        decision, prop_valid = pac_object.decide_pac(examples[:i], query, VALIDITY)
        tec = time.perf_counter()
        current_times.append(tec - tac)
    print(f"Processed {NO_OF_QUERIES} queries for {i} variables in {sum(current_times):0.1f} seconds")
    running_times.append(sum(current_times) / len(current_times))
#plt.plot(range(1, len(running_times) + 1), running_times)
plt.plot(range(0, 90000, 5000), running_times)
plt.xlabel('Number of examples')
plt.ylabel('Running time (s)')
plt.title(f"10 queries")
plt.savefig(f"running_times.png")
plt.show()
tac = time.perf_counter()
print(f"The entire process took {tac - tic:0.1f} seconds.")
