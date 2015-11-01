__author__ = 'ctiwary'
import numpy as np
from sklearn.feature_selection import VarianceThreshold

# http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection
# user guide http://scikit-learn.org/stable/modules/feature_selection.html#feature-selection

# feature selection based on VarianceThreshold
from sklearn.feature_selection import VarianceThreshold
X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
print np.shape(X)
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
features_list = sel.fit_transform(X)
print features_list
print sel.inverse_transform(features_list)
print sel.get_support()
print dir(sel)

# Univariate feature selection

from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
iris = load_iris()
X, y = iris.data, iris.target
print X.shape
X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
print X_new.shape
