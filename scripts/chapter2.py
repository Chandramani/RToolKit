import numpy as np
import pandas as pd
import sys
import itertools
from collections import Counter
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.style.use('ggplot')


class Chapter2:

    def __init__(self, mode="prod"):
        self.print_message = False
        if mode == "debug":
            self.print_message = True

    def log_message(self, message=""):
        if self.print_message:
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
        :rtype : object
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
        return per_non_zero

    @staticmethod
    def group_data_frame(df=pd.DataFrame(), metrics_to_group=[], metrics_to_group_by=[], operation="sum"):
        """
        :rtype : DataFrame
        :param df:
        :param metrics_to_group:
        :param metrics_to_group_by:
        :param operation:
        :return:
        """
        valid_operations = ["sum", "mean", "median", "max", "min", "prod", "count_nonzero", "count"]
        if "all_numeric" in metrics_to_group:
            if operation in valid_operations and operation != "count_nonzero":
                # col_list = metrics_to_group + metrics_to_group_by
                grouped_frame = df.groupby(metrics_to_group_by).agg(operation).reset_index()
                return grouped_frame
            elif operation == "count_nonzero":
                # col_list = metrics_to_group + metrics_to_group_by
                grouped_frame = df.groupby(metrics_to_group_by).agg(np.count_nonzero).reset_index()
                return grouped_frame
            else:
                # FIXME raise Exception
                print "incorrect input for argument operation, valid option is any one of these = ", valid_operations
                sys.exit(1)
        else:
            if operation in valid_operations and operation != "count_nonzero":
                col_list = metrics_to_group + metrics_to_group_by
                grouped_frame = df[col_list].groupby(metrics_to_group_by).agg(operation).reset_index()
                return grouped_frame
            elif operation == "count_nonzero":
                col_list = metrics_to_group + metrics_to_group_by
                grouped_frame = df[col_list].groupby(metrics_to_group_by).agg(np.count_nonzero).reset_index()
                return grouped_frame
            else:
                # FIXME raise Exception
                print "incorrect input for argument operation, valid option is any one of these = ", valid_operations
                sys.exit(1)


    @staticmethod
    def get_most_common_value(lst):
        data = Counter(lst)
        return data.most_common(1)

    @staticmethod
    def categorical_variable_count(input_col):
        data = Counter(input_col)
        week_count_frame = pd.DataFrame(columns=['count'])
        for val in data:
            week_count_frame.loc[val] = data[val]
        return week_count_frame

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

    @staticmethod
    def derive_variance(input_col):
        input_col.var()

    def get_count_of_week_by_account(self, input_data_frame, account_col=[], week_col=[]):
        account_week_count = self.group_data_frame(metrics_to_group=week_col, metrics_to_group_by=account_col,
                                                   df=input_data_frame, operation="count")
        return account_week_count

    @staticmethod
    def get_days_for_accounts(input_data_frame, account_col="", week_col=""):
        """

        :rtype : DataFrame
        """
        usage_grouped = input_data_frame[[account_col, week_col]].groupby([account_col])
        usage_age_account = pd.DataFrame(usage_grouped[week_col].apply(lambda x: (x.max() - x.min())).reset_index())
        usage_age_account['days'] = (usage_age_account['week'] / np.timedelta64(1, 'D')).astype(int)
        return usage_age_account

    @staticmethod
    def get_months_for_account(input_data_frame, account_col="", week_col=""):
        """

        :rtype : DataFrame
        """
        usage_grouped = input_data_frame[[account_col, week_col]].groupby([account_col])
        usage_age_account = pd.DataFrame(usage_grouped[week_col].apply(lambda x: (x.max() - x.min())/30).reset_index())
        usage_age_account['months'] = (usage_age_account['week'] / np.timedelta64(1, 'D')).astype(int)
        return usage_age_account[[account_col, 'months']]

    @staticmethod
    def filter_accounts_by_col_value(input_data_frame, col_name, col_value, operator="eq"):
        """
        :rtype : DataFrame
        :param input_data_frame:
        :param col_name:
        :param col_value:
        :param operator: values eq|lt|gt|lte|gte|ne
        :return:
        """
        if operator == "gt":
            return input_data_frame[input_data_frame[col_name] > col_value]
        elif operator == "lt":
            return input_data_frame[input_data_frame[col_name] < col_value]
        elif operator == "eq":
            return input_data_frame[input_data_frame[col_name] == col_value]
        elif operator == "gte":
            return input_data_frame[input_data_frame[col_name] >= col_value]
        elif operator == "lte":
            return input_data_frame[input_data_frame[col_name] <= col_value]
        elif operator == "ne":
            return input_data_frame[input_data_frame[col_name] != col_value]
        else:
            print "invalid operator"
            sys.exit(1)

    def create_summary_report(self, input_data_frame, columns_to_ignore=[]):
        """
        :rtype : DataFrame
        :return: Data Frame with column name as index
        """
        no_of_accounts = input_data_frame['account_id'].nunique()
        summary_frame = pd.DataFrame(columns=(
            'sparsity_flag', 'sparse_percentage', 'per_distinct', 'count_of_distinct', 'min', 'first_quartile', 'mean',
            'median', 'third_quartile', 'max', 'standard_deviation', 'variance', 'per_zero_accounts_sum'))
        for col in input_data_frame.columns.values:
            if col not in columns_to_ignore:
                sparsity_flag, sparse_per = self.check_col_sparsity(input_data_col=input_data_frame[col],
                                                                    sparsity_threshold=0.5,
                                                                    col_name=col)
                per_distinct, count_of_distinct = self.check_col_count_unique(input_data_col=input_data_frame[col],
                                                                              col_name=col)
                min_val = input_data_frame[col].min()
                first_quartile = np.percentile(input_data_frame[col], 25.0)
                mean = input_data_frame[col].mean()
                median = input_data_frame[col].median()
                third_quartile = np.percentile(input_data_frame[col], 75.0)
                max_val = input_data_frame[col].max()
                std = input_data_frame[col].std()
                variance = input_data_frame[col].var()
                per_zero_accounts_sum = self.get_per_of_acct_zero_sum(input_data_frame=input_data_frame[['account_id', col]], input_col=col, account_col='account_id', total_no_of_accounts=no_of_accounts)
                summary_frame.loc[col] = [sparsity_flag, sparse_per, per_distinct, count_of_distinct, min_val,
                                          first_quartile, mean, median, third_quartile, max_val, std, variance,
                                          per_zero_accounts_sum]
        return summary_frame

    @staticmethod
    def example_cat_var_count_per():
        s = pd.Series(["a", "b", "c", "a", "a", "b", "c", "a", "a", "b", "c", "a"], dtype="category")
        cat_var_count = Chapter2.categorical_variable_count(s)
        print cat_var_count
        length = len(s)
        for key in cat_var_count.keys():
            print key, cat_var_count[key], (cat_var_count[key] / float(length)) * 100

    @staticmethod
    def plot_hist(input_data_frame, bins=10):
        input_data_frame.hist(bins=bins)

    def get_accounts_by_sum(self, input_data_frame=pd.DataFrame(), input_col="", account_col=""):
        col_sum = self.group_data_frame(df=input_data_frame, metrics_to_group=[input_col], metrics_to_group_by=[account_col], operation="sum")
        return col_sum

    def get_accounts_by_frequency(self, input_data_frame=pd.DataFrame(), input_col="", account_col=""):
        col_frequency = self.group_data_frame(df=input_data_frame, metrics_to_group=[input_col], metrics_to_group_by=[account_col], operation="count")
        return col_frequency

    def get_per_of_acct_zero_sum(self, input_data_frame=pd.DataFrame(), input_col="", account_col="", total_no_of_accounts=int):
        col_sum = self.get_accounts_by_sum(input_data_frame, input_col, account_col)
        col_sum = col_sum[col_sum[input_col] == 0]
        return (len(col_sum)/float(total_no_of_accounts))*100

# also add feature usage coverage
if __name__ == '__main__':
    obj = Chapter2(mode="prod")
    input_data = pd.read_csv("../data/UsageData.csv")
    input_data['week'] = pd.to_datetime(input_data['week'])
    summary_report = obj.create_summary_report(input_data_frame=input_data, columns_to_ignore=input_data.columns[0:2])
    summary_report.to_csv("../data_exploration/summary_report.csv")
    account_week_count_frame = obj.get_count_of_week_by_account(input_data_frame=input_data, account_col=['account_id'],
                                                                week_col=['week'])
    accounts_shortlisted_weeks = Chapter2.filter_accounts_by_col_value(input_data_frame=account_week_count_frame,
                                                                       col_name="week", col_value=39, operator="gte")
    account_days_count_frame = Chapter2.get_days_for_accounts(input_data_frame=input_data, account_col="account_id",
                                                              week_col="week")
    accounts_shortlisted_days = Chapter2.filter_accounts_by_col_value(input_data_frame=account_days_count_frame,
                                                                      col_name="days", col_value=39 * 7, operator="gte")
    account_month_count_frame = Chapter2.get_months_for_account(input_data_frame=input_data, account_col="account_id",
                                                                week_col="week")
    accounts_shortlisted_months = Chapter2.filter_accounts_by_col_value(input_data_frame=account_month_count_frame,
                                                                        col_name="months", col_value=5, operator="gte")
    Chapter2.plot_hist(account_week_count_frame)
    Chapter2.plot_hist(account_days_count_frame)
    Chapter2.plot_hist(account_month_count_frame, bins=5)
    week_count = Chapter2.categorical_variable_count(input_data['week'])
    week_count.plot()
    plt.show()
