import numpy as np
import pandas as pd
import sys
import itertools
import matplotlib.pyplot as plt

from collections import Counter
from Commons.Utils import Utils
from pandas.tools.plotting import scatter_matrix
from datetime import datetime

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
            return True, np.round(per_zero * 100)

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


    @staticmethod
    def get_per_most_common_value(lst):
        lst.fillna(0)
        data = Counter(lst)
        count_rows = len(lst)
        return (data.most_common(1)[0][1] / float(count_rows)) * 100, data.most_common(1)[0][0]

    @staticmethod
    def categorical_variable_count(input_col):
        data = Counter(input_col)
        count_frame = pd.DataFrame(columns=['count'])
        for val in data:
            count_frame.loc[val] = data[val]
        return count_frame

    @staticmethod
    def bivariate_categorical_to_categorical(input_col):
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
        account_week_count = Utils.group_data_frame(metrics_to_group=week_col, metrics_to_group_by=account_col,
                                                   df=input_data_frame, operation="count")
        return account_week_count

    def create_summary_report(self, input_data_frame, columns_to_ignore=[]):
        """
        :rtype : DataFrame
        :return: Data Frame with column name as index
        """
        no_of_accounts = input_data_frame[account_col].nunique()
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
                per_zero_accounts_sum = Utils.get_per_of_acct_zero_sum(
                    input_data_frame=input_data_frame[[account_col, col]], input_col=col, account_col=account_col,
                    total_no_of_accounts=no_of_accounts)
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


# also add feature usage coverage
if __name__ == '__main__':
    account_col = 'account_id'
    week_col = 'week'
    obj = Chapter2(mode="prod")
    usage_data = Utils.read_and_prepare_usage_data("../data/UsageData.csv")
    account_data = Utils.read_and_prepare_acct_data(path_to_input_file="../data/AccountsDataOriginal.csv")

    summary_report = obj.create_summary_report(input_data_frame=usage_data, columns_to_ignore=usage_data.columns[0:2])
    summary_report.to_csv("../data_exploration/summary_report.csv")
    account_week_count_frame = obj.get_count_of_week_by_account(input_data_frame=usage_data, account_col=[account_col],
                                                                week_col=[week_col])
    accounts_shortlisted_weeks = Utils.filter_accounts_by_col_value(input_data_frame=account_week_count_frame,
                                                                       col_name="week", col_value=39, operator="gte")
    account_days_count_frame = Utils.get_days_for_accounts(input_data_frame=usage_data, account_col="account_id",
                                                              week_col="week")
    accounts_shortlisted_days = Utils.filter_accounts_by_col_value(input_data_frame=account_days_count_frame,
                                                                      col_name="days", col_value=39 * 7, operator="gte")
    account_month_count_frame = Utils.get_months_for_account(input_data_frame=usage_data, account_col="account_id",
                                                                week_col="week")
    accounts_shortlisted_months = Utils.filter_accounts_by_col_value(input_data_frame=account_month_count_frame,
                                                                        col_name="months", col_value=5, operator="gte")
    Utils.create_and_save_correlation_covariance_matrix(usage_data[3:usage_data.shape[1]],
                                                           correlation_file_path="../data_exploration/correlation_matrix.csv",
                                                           covariance_file_path="../data_exploration/covariance_matrix.csv")

    print Chapter2.categorical_variable_count(account_data['status'][account_data[account_col].isin(
        list(set.intersection(set(usage_data[account_col]), set(account_data[account_col]))))])
    for col in usage_data.columns.values:
        if col not in usage_data.columns[0:2]:
            print col
            agg_data = usage_data[[account_col, col]].groupby(account_col).sum().reset_index()
            print Chapter2.categorical_variable_count(
                account_data['status'][account_data[account_col].isin(agg_data[account_col][agg_data[col] <= 0])])

    usage_data.drop('total_softs_deleted_user_sessions', axis=1, inplace=True)
    # np.set_printoptions(precision=15)
    # scatter_matrix(usage_data[14:16], alpha=0.2, figsize=(6, 6), diagonal='kde')
    # plt.show()
    print Chapter2.categorical_variable_count(input_col=account_data['industry'])
    cross_tab_industry_status = pd.crosstab(account_data['status'], account_data['industry']).apply(
        lambda x: (x / float(x.sum())) * 100, axis=0)
    usage_data_agg = Utils.group_data_frame(df=usage_data, metrics_to_group="all_numeric",
                                               metrics_to_group_by=[account_col], operation="sum")
    usage_data_agg_status = pd.merge(left=usage_data_agg, right=account_data, on=account_col)
    color_groups = usage_data_agg_status.groupby('status')
    # Plot
    fig, ax = plt.subplots()
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
    for name, group in color_groups:
        ax.plot(group.logins, group.transactions, marker='o', linestyle='', ms=12, label=name)
    ax.legend()
    Utils.plot_hist(account_week_count_frame)
    Utils.plot_hist(account_days_count_frame)
    Utils.plot_hist(account_month_count_frame, bins=5)
    week_count = Chapter2.categorical_variable_count(usage_data[week_col])
    week_count.plot()
    usage_data_agg_status.loc[usage_data_agg_status.status=='active','statusID'] = 0
    usage_data_agg_status.loc[usage_data_agg_status.status=='churn','statusID'] = 1
    usage_data_agg_status.plot(kind='scatter', x='logins', y='transactions', c='statusID')
    plt.show()

