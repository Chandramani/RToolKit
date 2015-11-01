__author__ = 'ctiwary'

import pandas as pd

input_data = pd.read_csv("test.csv")

# if input_data['Name'] == "Aparajita":
#     print input_data['Character']

for name in input_data['Name']:
    if name == 'Aparajita':
        print name, "acchi baachi"
    else:
        print name,"duusta"