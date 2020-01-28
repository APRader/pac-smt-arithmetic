import pandas as pd
from z3 import *
import pac

ambient, coolant, u_d, u_q, pm = Reals('ambient coolant u_d u_q pm')

df = pd.read_csv('pmsm_temperature_data.csv')
# Kaggle task says to drop torque, since it is not reliably measurable.
# But PAC can deal with imperfect data, so we will use it anyway.
target_features = ['pm', 'stator_tooth', 'stator_yoke', 'stator_winding']

set_option(rational_to_decimal=True)
profile_4 = df.loc[df['profile_id'] == 4]
print(profile_4.size)
confidence = 0.9
number_of_examples = pac.sample_size(confidence, 0.01)
print(number_of_examples)
example = lambda i: And(ambient > df.loc[5 * i:5 * i + 4, 'ambient'].min(),
                        ambient < df.loc[5 * i:5 * i + 4, 'ambient'].max(),
                        coolant > df.loc[5 * i:5 * i + 4, 'coolant'].min(),
                        coolant < df.loc[5 * i:5 * i + 4, 'coolant'].max(),
                        u_d > df.loc[5 * i:5 * i + 4, 'u_d'].min(), u_d < df.loc[5 * i:5 * i + 4, 'u_d'].max(),
                        u_q > df.loc[5 * i:5 * i + 4, 'u_q'].min(), u_q < df.loc[5 * i:5 * i + 4, 'u_q'].max(),
                        pm > df.loc[5 * i:5 * i + 4, 'pm'].min(), pm < df.loc[5 * i:5 * i + 4, 'pm'].max(), )

examples = [example(i) for i in range(number_of_examples)]
#print(examples)
query = pm < -1.55
print(pac.decide_pac(ambient > -9, examples, query, 0.95))