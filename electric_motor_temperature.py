import pandas as pd
from z3 import *
import pac

ambient, coolant, u_d, u_q, motor_speed, torque, i_d, i_q, pm, stator_yoke, stator_tooth, stator_winding = \
    Reals('ambient coolant u_d u_q motor_speed torque i_d i_q pm stator_yoke stator_tooth stator_winding')

z3_vars = [ambient, coolant, u_d, u_q, motor_speed, torque, i_d, i_q, pm, stator_yoke, stator_tooth, stator_winding]

df = pd.read_csv('pmsm_temperature_data.csv')
# Kaggle task says to drop torque, since it is not reliably measurable.
# But PAC can deal with imperfect data, so we will use it anyway.
target_features = ['pm', 'stator_tooth', 'stator_yoke', 'stator_winding']

set_option(rational_to_decimal=True)
profile_4 = df.loc[df['profile_id'] == 4]
#print(df.size)
confidence = 0.9
number_of_examples = pac.sample_size(confidence, 0.01)
#print(number_of_examples)
examples = pac.create_examples(profile_4, z3_vars, 2)

# print(examples)
query = pm < -1.55
print(pac.decide_pac(ambient > -9, examples, query, 0.95))
