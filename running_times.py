import motor_util
import time
import pac
import pandas as pd
import matplotlib.pyplot as plt

KB_COMPRESSION = 1000000  # compression factor for knowledge base, larger than amount of data points in any profile
EXAMPLE_COMPRESSION = 100
VALIDITY = 0.5
z3_vars = motor_util.set_up_variables()

# columns: number of variables, rows: different combinations of variables
queries = [z3_vars.get("ambient") > 0, z3_vars.get("ambient") + z3_vars.get("coolant") > 0,
           z3_vars.get("ambient") + z3_vars.get("coolant") - z3_vars.get("u_d") > 0,
           z3_vars.get("ambient") + z3_vars.get("coolant") - z3_vars.get("u_d") + z3_vars.get("u_q") > 0,
           z3_vars.get("ambient") + z3_vars.get("coolant") - z3_vars.get("u_d") + z3_vars.get("u_q")
           - z3_vars.get("motor_speed") > 0,
           z3_vars.get("ambient") + z3_vars.get("coolant") - z3_vars.get("u_d") + z3_vars.get("u_q")
           - z3_vars.get("motor_speed") + z3_vars.get("torque") > 0,
           z3_vars.get("ambient") + z3_vars.get("coolant") - z3_vars.get("u_d") + z3_vars.get("u_q")
           - z3_vars.get("motor_speed") + z3_vars.get("torque") - z3_vars.get("i_d") > 0,
           z3_vars.get("ambient") + z3_vars.get("coolant") - z3_vars.get("u_d") + z3_vars.get("u_q")
           - z3_vars.get("motor_speed") + z3_vars.get("torque") - z3_vars.get("i_d") + z3_vars.get("i_q") > 0,
           z3_vars.get("ambient") + z3_vars.get("coolant") - z3_vars.get("u_d") + z3_vars.get("u_q")
           - z3_vars.get("motor_speed") + z3_vars.get("torque") - z3_vars.get("i_d") + z3_vars.get("i_q")
           - z3_vars.get("pm") > 0
           ]

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
tuc = time.perf_counter()
print(f"Created inequalities in {tuc - toc:0.1f} seconds.")

running_times = []
for query in queries:
    tuc = time.perf_counter()
    decision, prop_valid = pac_object.decide_pac(examples, query, VALIDITY)
    tac = time.perf_counter()
    running_times.append(tac - tuc)
    print(f"PAC decided to {decision} since {prop_valid:0.3f} of the examples were valid "
          f"after thinking for {tac - tuc:0.1f} seconds.")

plt.plot(range(1, len(running_times) + 1), running_times)
plt.xlabel('Number of arguments')
plt.ylabel('Running time (s)')
plt.title(f"{len(examples)} examples")
plt.savefig(f"running_times.png")
plt.show()
tac = time.perf_counter()
print(f"The entire process took {tac - tic:0.1f} seconds.")