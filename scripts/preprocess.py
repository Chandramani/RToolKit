__author__ = 'ctiwary'

import pandas as pd
from string import maketrans

data = pd.read_csv("../data/UsageData_bkp.csv")

# input_data = pd.read_csv("../data/UsageData.csv")
# print input_data.head()
# lower_case_column_name = [x.lower() for x in input_data.columns.values]
# Chapter2.log_message(lower_case_column_name)
# input_data.columns = lower_case_column_name
# Chapter2.log_message(message=input_data.columns.values)
# print input_data.head()
# input_data.to_csv("../data/UsageData.csv", index=False)
# Chapter2.log_message(input_data.columns.values)

print data['account_id'].head()

print data['account_id'].nunique()

data['account_id'] = data['account_id'].str[10:18].str.lower()

print data['account_id'].nunique()

intab = "abcdefghijklmnopqrstuvwxyz1234567890"
outtab = "1234567890abcdefghijklmnopqrstuvwxyz"
trantab = maketrans(intab, outtab)

data['account_id'] = data['account_id'].str.translate(trantab)

data.to_csv("../data/UsageDataMasked.csv",index=False)
#print list(set(data['account_id']) - set(data['account_id'].str.translate(trantab)))

