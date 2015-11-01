__author__ = 'ctiwary'

import pandas as pd
import sys
import numpy as np


class Utils:

    def __init__(self, mode="prod"):
        self.print_message = False
        if mode == "debug":
            self.print_message = True

    account_col = 'account_id'
    week_col = 'week'

    @staticmethod
    def get_accounts_non_zero_sum(input_data_frame=pd.DataFrame(), account_col=""):
        col_sum = input_data_frame.groupby(by=account_col).sum()
        col_sum = col_sum.sum(axis=1)
        col_sum.fillna(0, inplace=True)
        col_sum_none_zero = col_sum[col_sum > 0]
        col_sum_zero = col_sum[col_sum == 0]
        return col_sum_none_zero.index, col_sum_zero.index

    @staticmethod
    def read_and_prepare_usage_data(path_to_input_file="../data/UsageData.csv"):
        input_data = pd.read_csv(path_to_input_file)
        input_data[Utils.week_col] = pd.to_datetime(input_data[Utils.week_col])
        input_data.fillna(0, inplace=True)
        account_non_zero, account_zero = Utils.get_accounts_non_zero_sum(
            input_data_frame=input_data.drop(Utils.week_col, axis=1), account_col=Utils.account_col)
        input_data = input_data[input_data[Utils.account_col].isin(account_non_zero)]
        return input_data


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
    def get_accounts_by_sum(input_data_frame=pd.DataFrame(), input_col="", account_col=""):
        col_sum = Utils.group_data_frame(df=input_data_frame, metrics_to_group=[input_col],
                                            metrics_to_group_by=[account_col], operation="sum")
        return col_sum

    @staticmethod
    def get_accounts_by_frequency(input_data_frame=pd.DataFrame(), input_col="", account_col=""):
        col_frequency = Utils.group_data_frame(df=input_data_frame, metrics_to_group=[input_col],
                                                  metrics_to_group_by=[account_col], operation="count")
        return col_frequency

    @staticmethod
    def get_per_of_acct_zero_sum(input_data_frame=pd.DataFrame(), input_col="", account_col="",
                                 total_no_of_accounts=int):
        col_sum = Utils.get_accounts_by_sum(input_data_frame, input_col, account_col)
        col_sum = col_sum[col_sum[input_col] == 0]
        return (len(col_sum) / float(total_no_of_accounts)) * 100

    @staticmethod
    def plot_hist(input_data_frame, bins=10):
        input_data_frame.hist(bins=bins)

    @staticmethod
    def create_and_save_correlation_covariance_matrix(input_data_frame, correlation_file_path, covariance_file_path):
        """
        :rtype : Null
        """
        correlation_matrix = input_data_frame[3:input_data_frame.shape[1]].corr(method='pearson', min_periods=1)
        covariance_matrix = input_data_frame[3:input_data_frame.shape[1]].cov(min_periods=1)
        correlation_matrix.to_csv(correlation_file_path, index=False)
        covariance_matrix.to_csv(covariance_file_path, index=False)

    @staticmethod
    def read_and_prepare_acct_data(path_to_input_file="../data/AccountsDataOriginal.csv"):
        input_data = pd.read_csv(path_to_input_file)
        input_data.fillna(0, inplace=True)
        input_data.loc[input_data.no_of_users <= 0, 'subscription_value'] = 0
        input_data.loc[input_data.no_of_users < 0, 'no_of_users'] = 0
        input_data.loc[input_data.no_of_users <= 0, 'status'] = 'churn'
        input_data.loc[input_data.no_of_users > 0, 'status'] = 'active'
        return input_data

    @staticmethod
    def get_days_for_accounts(input_data_frame, account_col="", week_col=""):
        """

        :rtype : DataFrame
        """
        usage_grouped = input_data_frame[[account_col, week_col]].groupby([account_col])
        usage_age_account = pd.DataFrame(usage_grouped[week_col].apply(lambda x: (x.max() - x.min())).reset_index())
        usage_age_account['days'] = (usage_age_account[week_col] / np.timedelta64(1, 'D')).astype(int)
        return usage_age_account

    @staticmethod
    def get_months_for_account(input_data_frame, account_col="", week_col=""):
        """

        :rtype : DataFrame
        """
        usage_grouped = input_data_frame[[account_col, week_col]].groupby([account_col])
        usage_age_account = pd.DataFrame(
            usage_grouped[week_col].apply(lambda x: (x.max() - x.min()) / 30).reset_index())
        usage_age_account['months'] = (usage_age_account[week_col] / np.timedelta64(1, 'D')).astype(int)
        return usage_age_account[[account_col, 'months']]
