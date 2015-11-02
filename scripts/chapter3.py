__author__ = 'ctiwary'

import numpy as np
import pandas as pd
from Commons.Utils import Utils
import pylab
import sys

import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt


class Chapter3:
    def __init__(self):
        pass

    # This method can be used in production directly. It is also general enough to be included in every project
    @staticmethod
    def limitTransactionsWithinWindow(usageDF, windowToSkip, windowToInclude):
        # filter out any transaction less or equal than windowToSkip and anything greater than window to include
        return usageDF.loc[np.logical_and(usageDF.weekID > windowToSkip, usageDF.weekID <= windowToInclude), :]

    # This method can be used in production directly. But this is customized for every project
    # It is used to generate the usage-based features
    @staticmethod
    def generate_usage_features(tenant_transaction_window_lt,
                                tenant_transaction_window_st1, tenant_transaction_window_st2,
                                sumFields, maxFields, avgFields, sumFeatures, maxFeatures,
                                avgFeatures, avgFeaturesChange):
        lt_window_size = max(tenant_transaction_window_lt.weekID) - min(tenant_transaction_window_lt.weekID) + 1
        st1_window_size = max(tenant_transaction_window_st1.weekID) - min(tenant_transaction_window_st1.weekID) + 1
        st2_window_size = max(tenant_transaction_window_st2.weekID) - min(tenant_transaction_window_st2.weekID) + 1
        df_avg = tenant_transaction_window_lt[avgFields].groupby(level=0).agg(
            np.sum) / lt_window_size  # used a normalized sum instead of mean because it assumes that the missing transactions have zero values rather than do not exist
        df_max = tenant_transaction_window_lt[maxFields].groupby(level=0).agg(np.max)
        df_sum = tenant_transaction_window_lt[sumFields].groupby(level=0).agg(np.sum)
        # compute the avg variables for the short term windows and compute the change
        df_avg_st1 = (tenant_transaction_window_st1[avgFields].groupby(level=0).agg(np.sum) / st1_window_size) + 0.001
        df_avg_st2 = (tenant_transaction_window_st2[avgFields].groupby(level=0).agg(np.sum) / st2_window_size) + 0.001
        df_change = np.log10(df_avg_st1.div(df_avg_st2,
                                            fill_value=0.001))  # I am using log change instead of percent change. It behaves much better. I am also using div instead of division so that I can fill missing values before dividing
        # I assumed above that all the fields can not have negative values. This needs to be checked first.
        # change the name of the features
        df_avg.columns = avgFeatures
        df_sum.columns = sumFeatures
        df_max.columns = maxFeatures
        df_change.columns = avgFeaturesChange
        # Now we will change the names of the columns and return the concatenated data frame
        return pd.concat([df_avg, df_max, df_sum, df_change], axis=1)

    @staticmethod
    def add_sufix_to_col(col_list,suffix=""):
        name_list = []
        for var in list(col_list):
            name_list.append(var+str(suffix))
        return name_list

    @staticmethod
    def check_col_sparsity( input_data_col="", sparsity_threshold=0.5, col_name=""):
        """
        :param input_data_col: input data column of data frame
        :param sparsity_threshold: the threshold of sparsity
        :return:
        """
        count_of_rows = len(input_data_col)
        input_data_col = input_data_col.replace(0, np.nan)
        count_of_non_zero = input_data_col.count()
        per_zero = (count_of_rows - count_of_non_zero) / float(count_of_rows)
        if per_zero < sparsity_threshold:
            return False, np.round(per_zero * 100)
        else:
            return True, np.round(per_zero * 100)

    @staticmethod
    def check_col_count_unique( input_data_col="", col_name=""):
        """
        This function checks for sparsity of columns
        :rtype : object
        :param input_data_col:
        :return:
        """
        count_of_rows = len(input_data_col)
        input_data_col = input_data_col.replace(0, np.nan)
        count_of_distinct = input_data_col.nunique()
        per_distinct = count_of_distinct / float(count_of_rows)
        return per_distinct * 100, count_of_distinct

    @staticmethod
    def create_summary_report_features( input_data_frame, columns_to_ignore=[]):
        """
        :rtype : DataFrame
        :return: Data Frame with column name as index
        """
        no_of_accounts = len(input_data_frame)
        summary_frame = pd.DataFrame(columns=(
            'sparsity_flag', 'sparse_percentage', 'per_distinct', 'count_of_distinct', 'min', 'first_quartile', 'mean',
            'median', 'third_quartile', 'max', 'standard_deviation', 'variance'))
        for col in input_data_frame.columns.values:
            if col not in columns_to_ignore:
                sparsity_flag, sparse_per = Chapter3.check_col_sparsity(input_data_col=input_data_frame[col],sparsity_threshold=0.5,col_name=col)
                per_distinct, count_of_distinct = Chapter3.check_col_count_unique(input_data_col=input_data_frame[col],col_name=col)
                min_val = input_data_frame[col].min()
                first_quartile = np.percentile(input_data_frame[col], 25.0)
                mean = input_data_frame[col].mean()
                median = input_data_frame[col].median()
                third_quartile = np.percentile(input_data_frame[col], 75.0)
                max_val = input_data_frame[col].max()
                std = input_data_frame[col].std()
                variance = input_data_frame[col].var()
                summary_frame.loc[col] = [sparsity_flag, sparse_per, per_distinct, count_of_distinct, min_val,
                                          first_quartile, mean, median, third_quartile, max_val, std, variance
                                          ]
        return summary_frame


if __name__ == '__main__':
    account_col = 'account_id'
    week_col = 'week'
    status_label = 'status'
    status_active = 'active'
    status_churn = 'churn'
    usage_data = Utils.read_and_prepare_usage_data("../data/UsageData.csv")
    account_data = Utils.read_and_prepare_acct_data(path_to_input_file="../data/AccountsDataOriginal.csv")

    usage_data_dates = usage_data.groupby([account_col])[week_col].agg(
        {'First_Used': np.min, 'Last_Used': np.max}).reset_index()
    print len(usage_data)
    usage_data = usage_data.groupby([account_col, week_col]).sum().reset_index()
    print len(usage_data)
    usage_data = pd.merge(usage_data, usage_data_dates, on=account_col)
    print len(usage_data)
    usage_data.loc[:, 'weekID'] = (usage_data.Last_Used - usage_data[week_col]) / np.timedelta64(1, 'W')
    print usage_data[[account_col, week_col, 'weekID']].head()

    windowToSkip = 12  # skip from weekID = 0 to weekID = 7
    windowToInclude = 21  # skip anything beyond week 15
    usage_transaction_window_lt = Chapter3.limitTransactionsWithinWindow(usage_data, windowToSkip, windowToInclude)
    # Short term window1 (skip 8, then include the next 4)
    windowToInclude = 17
    usage_transaction_window_st1 = Chapter3.limitTransactionsWithinWindow(usage_data, windowToSkip, windowToInclude)
    windowToSkip = 17
    windowToInclude = 21
    usage_transaction_window_st2 = Chapter3.limitTransactionsWithinWindow(usage_data, windowToSkip, windowToInclude)


    avgFields = ['logins', 'forms_filled', 'links_clicked', 'transactions', 'trans_succ', 'trans_failed',
     'new_user_sessions', 'user_sessions', 'total_known_user_sessions', 'total_unknown_user_sessions',
     'known_user_sessions', 'anon_user_sessions', 'known_user_sessions_net', 'total_softs_deleted_user_sessions',
     'anonwebvisitcount__c']

    usage_transaction_window_lt['usage_breadth'] =  (usage_transaction_window_lt[avgFields]>0).sum(axis=1)
    usage_transaction_window_st1['usage_breadth'] =  (usage_transaction_window_st1[avgFields]>0).sum(axis=1)
    usage_transaction_window_st2['usage_breadth'] =  (usage_transaction_window_st2[avgFields]>0).sum(axis=1)
    usage_transaction_window_lt['usage_breadth'].head()

    avgFields.append('usage_breadth')

    avgFeatures = ['logins_avg', 'forms_filled_avg', 'links_clicked_avg', 'transactions_avg', 'trans_succ_avg', 'trans_failed_avg',
     'new_user_sessions_avg', 'user_sessions_avg', 'total_known_user_sessions_avg', 'total_unknown_user_sessions_avg',
     'known_user_sessions_avg', 'anon_user_sessions_avg', 'known_user_sessions_net_avg', 'total_softs_deleted_user_sessions_avg',
     'anonwebvisitcount__c_avg', 'usage_breadth_avg']

    avgFeaturesChange = ['logins_change', 'forms_filled_change', 'links_clicked_change', 'transactions_change', 'trans_succ_change', 'trans_failed_change',
     'new_user_sessions_change', 'user_sessions_change', 'total_known_user_sessions_change', 'total_unknown_user_sessions_change',
     'known_user_sessions_change', 'anon_user_sessions_change', 'known_user_sessions_net_change', 'total_softs_deleted_user_sessions_change',
     'anonwebvisitcount__c_change', 'usage_breadth_change']

    sumFields = []
    maxFields = []
    sumFeatures = []
    maxFeatures = []
    
    usage_transaction_window_lt.set_index(account_col,inplace=True)
    usage_transaction_window_st1.set_index(account_col,inplace=True)
    usage_transaction_window_st2.set_index(account_col,inplace=True)
    
    df_usage = Chapter3.generate_usage_features(usage_transaction_window_lt, usage_transaction_window_st1, 
                   usage_transaction_window_st2, sumFields, maxFields, avgFields, sumFeatures, maxFeatures,
                                  avgFeatures, avgFeaturesChange)
    # derive frequency features
    avgFields.remove('usage_breadth')
    usage_frq = usage_transaction_window_lt.groupby(level=0).agg(np.count_nonzero)
    usage_frq = usage_frq[avgFields]
    usage_frq.columns = Chapter3.add_sufix_to_col(usage_frq,suffix="_freq")

    features_final = pd.merge(df_usage,usage_frq, right_index=True, left_index=True)

    names = list(features_final.columns.values)
    # names.remove('InferredChurn')

    correlation_matrix = features_final.corr()
    fig, ax = plt.subplots(figsize=(15, 15))
    sm.graphics.plot_corr(correlation_matrix, xnames=names,ax=ax)
    # plt.show()
    pylab.savefig('../images/correlation_matrix.png')

    account_data.set_index(account_col, inplace=True)
    feature_summary = Chapter3.create_summary_report_features(features_final, columns_to_ignore=[])

    features_final = pd.merge(features_final, account_data[[status_label,'age_in_months']], left_index=True, right_index=True)

    features_final_active = features_final[features_final[status_label] == status_active]
    features_final_churn = features_final[features_final[status_label] == status_churn]
    features_final_active.drop(status_label, inplace=True, axis=1)
    features_final_churn.drop(status_label, inplace=True, axis=1)
    feature_summary_active = Chapter3.create_summary_report_features(features_final_active, columns_to_ignore=[])
    feature_summary_churn = Chapter3.create_summary_report_features(features_final_churn, columns_to_ignore=[])
    feature_summary.to_csv("../data_exploration/chapter3_summary_report_feature_final.csv")

    feature_summary_active.to_csv("../data_exploration/chapter3_summary_report_feature_final_active.csv")
    feature_summary_churn.to_csv("../data_exploration/chapter3_summary_report_feature_final_churn.csv")

    feats = features_final.columns.values

    features_final.to_csv('../data/features_final.csv')

    for i in range(0, len(features_final.columns.values)):
        try:
            forActive = pd.Series(features_final.loc[features_final[status_label] == status_active, feats[i]]).values
            forChurned = pd.Series(features_final.loc[features_final[status_label] == status_churn, feats[i]]).values
            # There's no way in the current pipeline that there could be nulls at this point, but to make a final check
            print feats[i], ":", sum(pd.isnull(forActive)), "nulls for active;", sum(pd.isnull(forChurned)), "nulls for churned"
            plt.figure(i)
            plt.boxplot([forActive, forChurned])
            plt.title(feats[i])
            plt.close()
            pylab.savefig("../images/"+feats[i]+".png")
        except Exception, e: #not everything will be able to be plotted
            print 'Unable to plot boxplot for', feats[i], e