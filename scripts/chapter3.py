__author__ = 'ctiwary'

from Commons.Utils import Utils

class Chapter3:

    def __index__(self):
        pass

if __name__ == '__main__':
    account_col = 'account_id'
    week_col = 'week'
    usage_data = Utils.read_and_prepare_usage_data("../data/UsageData.csv")
    print usage_data.head()