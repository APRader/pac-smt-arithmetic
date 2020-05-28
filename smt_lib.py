from z3 import *
import pac
import numpy as np

NO_OF_EXAMPLES = 10

f = open("bmwlin_20_5_1.inter.bmc_k100.smt2", "r")
lines = f.readlines()

variables = []

for line in lines:
    if line.startswith("(declare-fun"):
        var_name = line.split()[1]
        type_name = line.split()[3]
        if type_name == "Bool)":
            variables.append(Bool(var_name))
        if type_name == "Real)":
            variables.append(Real(var_name))

f.close()

f = open("assertions.smt2", "r")
query = f.read()
f.close()

knowledge_base = [And(-1 <= variable, variable <= 1) for variable in variables if is_real(variable)]

pac_learner = pac.PACLearner(variables, knowledge_base)

no_of_real_variables = sum(list(map(is_real, variables)))
no_of_bool_variables = sum(list(map(is_bool, variables)))
bool_samples = [True, False]
examples = []

for _ in range(NO_OF_EXAMPLES):
    random_reals = np.random.uniform(-1, 1, no_of_real_variables)
    random_bools = np.random.choice(bool_samples, no_of_bool_variables)

