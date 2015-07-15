import numpy as np
import pandas as pd
import itertools
from collections import Counter


class Chapter2:
    def __init__(self):
        pass

    @staticmethod
    def log_message(message=""):
        print message

    def check_col_sparsity(self, input_data_col="", sparsity_threshold=0.5, col_name=""):
        """
        :param input_data_col: input data column of data frame
        :param sparsity_threshold: the threshold of sparsity
        :return:
        """
        self.log_message(message="inside function Checking column sparsity for columns = " + str(col_name))
        count_of_rows = len(input_data_col)
        input_data_col = input_data_col.replace(0, np.nan)
        count_of_non_zero = input_data_col.count()
        per_non_zero = count_of_non_zero / float(count_of_rows)
        if per_non_zero > sparsity_threshold:
            return False, np.round(per_non_zero * 100)
        else:
            return True, per_non_zero

    def check_col_count_unique(self, input_data_col="", col_name=""):
        """
        This function checks for sparsity of columns
        :param input_data_col:
        :return:
        """
        self.log_message(message="inside function calculating number of unique values for columns = " + str(col_name))
        count_of_rows = len(input_data_col)
        input_data_col = input_data_col.replace(0, np.nan)
        count_of_distinct = input_data_col.nunique()
        per_distinct = count_of_distinct / float(count_of_rows)
        return np.round(per_distinct * 100), count_of_distinct

    def per_most_common(self, input_data_col="", col_name=""):
        """
        :param input_data_col: input data column of data frame
        :param sparsity_threshold: the threshold of sparsity
        :return:
        """
        self.log_message(message="inside function calculating number of unique values for columns = " + str(col_name))
        count_of_rows = len(input_data_col)
        input_data_col = input_data_col.replace(0, np.nan)
        count_of_non_zero = input_data_col.count()
        per_non_zero = count_of_non_zero / float(count_of_rows)

    @staticmethod
    def get_most_common_value(lst):
        data = Counter(lst)
        return data.most_common(1)

    @staticmethod
    def categorical_variable_count(input_col):
        data = Counter(input_col)
        return data


if __name__ == '__main__':

    obj = Chapter2()
    input_data = pd.read_csv("../data/UsageData.csv")
    # print input_data.head()
    # lower_case_column_name = [x.lower() for x in input_data.columns.values]
    # Chapter2.log_message(lower_case_column_name)
    # input_data.columns = lower_case_column_name
    # Chapter2.log_message(message=input_data.columns.values)
    # print input_data.head()
    # input_data.to_csv("../data/UsageData.csv", index=False)
    # Chapter2.log_message(input_data.columns.values)
    columns_to_ignore = input_data.columns.values[0:2]
    for col in input_data.columns.values:
        if col not in columns_to_ignore:
            print obj.check_col_sparsity(input_data_col=input_data[col], sparsity_threshold=0.5, col_name=col)
            print obj.check_col_count_unique(input_data_col=input_data[col], col_name=col)
            print input_data[col].describe()
            print input_data[col].mean()
            print input_data[col].median()
            print input_data[col].std()
            # print Chapter2.get_most_common_value(lst=input_data[col])
            import sys
            sys.exit(0)
    s = pd.Series(["a","b","c","a","a","b","c","a","a","b","c","a"], dtype="category")
    cat_var_count = Chapter2.categorical_variable_count(s)
    length = len(s)
    for key in cat_var_count.keys():
        print key, cat_var_count[key], (cat_var_count[key]/float(length))*100