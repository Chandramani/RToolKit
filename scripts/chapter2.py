import numpy as np
import pandas as pd
import sys
import itertools
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.style.use('ggplot')


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
        per_zero = (count_of_rows - count_of_non_zero) / float(count_of_rows)
        if per_zero < sparsity_threshold:
            return False, np.round(per_zero * 100)
        else:
            return True, per_zero

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
        print per_distinct
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

    @staticmethod
    def derive_mean(input_col):
        input_col.mean()

    @staticmethod
    def derive_median(input_col):
        input_col.median()

    @staticmethod
    def derive_standard_deviation(input_col):
        input_col.std()

    @staticmethod
    def derive_standard_summary(input_col):
        input_col.describe()


if __name__ == '__main__':
    obj = Chapter2()
    input_data = pd.read_csv("../data/UsageData.csv")
    columns_to_ignore = input_data.columns.values[0:2]
    summary_frame = pd.DataFrame(columns=('sparsity_flag', 'sparse_percentage', 'per_distinct', 'count_of_distinct', 'mean', 'median','standard_deviation'))
    for col in input_data.columns.values:
        if col not in columns_to_ignore:
            sparsity_flag, sparse_per = obj.check_col_sparsity(input_data_col=input_data[col], sparsity_threshold=0.5, col_name=col)
            per_distinct, count_of_distinct = obj.check_col_count_unique(input_data_col=input_data[col], col_name=col)
            # print Chapter2.get_most_common_value(lst=input_data[col])
            # print input_data[col].describe()
            mean = input_data[col].mean()
            median = input_data[col].median()
            std = input_data[col].std()
            summary_frame.loc[col] = [sparsity_flag, sparse_per, per_distinct, count_of_distinct, mean, median, std]

    print summary_frame.head()
    summary_frame.to_csv("../data_exploration/summary_report.csv")
    s = pd.Series(["a","b","c","a","a","b","c","a","a","b","c","a"], dtype="category")
    cat_var_count = Chapter2.categorical_variable_count(s)
    length = len(s)
    for key in cat_var_count.keys():
        print key, cat_var_count[key], (cat_var_count[key]/float(length))*100

    # sum domain specific exploration
    print input_data['week'].min(), input_data['week'].max(), input_data['week'].nunique()
    print input_data['account_id'].nunique()
    week_account_id_count = input_data[['week', 'account_id']].groupby('week').count().reset_index()
    account_week_count = input_data[['week', 'account_id']].groupby('account_id').count().reset_index()
    print account_week_count['week'].describe()
    print week_account_id_count['account_id'].describe()
    print account_week_count.head()
    print week_account_id_count.head()
    week_account_id_count['week'] = pd.to_datetime(week_account_id_count['week'])
    week_account_id_count.set_index('week', inplace=True)
    print week_account_id_count.dtypes
    print week_account_id_count.head()
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax1.hist(account_week_count['week'], bins=20, range=(0, 400))
    ax2.hist(account_week_count['week'], bins=20, range=(0, 200))
    week_account_id_count.plot()
    plt.show()
