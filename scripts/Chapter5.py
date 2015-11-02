__author__ = 'ctiwary'

import numpy as np
import pandas as pd
from Commons.Utils import Utils
import pylab
import sys

import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, recall_score, precision_score


class Chapter5:
    def __init__(self):
        pass


    @staticmethod
    def build_model(featureDF, targetVariable, churnLabel, activeLabel):
        """ Build and test models using many trials with random training and test sets
        Parameters:
        -----------
        featureDF: The feature data frame, has one column for the target variable
        targetVariable: The name of the column containing the target variable
        churnLabel: The label of churn in the target variable (e.g. churn or 1)
        activeLabel: The label of active accounts in the target variable (e.g. active or 0)

        Returns:
        --------
        testScoresDF: a data frame containing the scores on different test sets, and the target variable
        Xtest: just the features of the last test set
        ytest: the target variables of the last test set
        scoreTest: the scores of the last test set.
        clf: the last model computed
        The Xtest,ytest and scoreTest will be used to inspect the results to see if they make sense
        clf will be used to check few things in the model (like importance)
        """

        # Since this is a generic ffuction, the safest thing to do is to exclude rows with NAs
        # But we should really deal with NAs before calling this function, because we know how the
        # data was generated.
        # I will assume for now that it is a two class problem, so only 2 classes
        x = featureDF.dropna(how='any')
        y = x.pop(targetVariable) # now x will not contain the target variable
        # What is the distribution of classes?
        print y.value_counts()
        # Now build the models
        ntrials=30
        plt.figure(figsize=(10,10))
        auc_list = list()
        active_scores, churn_scores = list(), list()
        clf = RandomForestClassifier(n_estimators=1000, max_depth= 3,
                                     oob_score=True, n_jobs=-1,
                                     class_weight='auto') #shallow trees, weights inversely proportional to class freqs
        for i in range(ntrials):
            Xtrain, Xtest, ytrain, ytest = train_test_split(x, y)
            clf.fit(Xtrain, ytrain)
            print "\n Iteration =",i
            print clf.oob_score_
            probas_ = clf.predict_proba(Xtest)
            class_label_ = clf.predict(Xtest)

            churn_scores.extend(probas_[np.where(ytest==churnLabel)[0],0])
            active_scores.extend(probas_[np.where(ytest==activeLabel)[0],0]);

            fpr, tpr, thresholds = roc_curve(ytest, probas_[:, 1], pos_label=churnLabel)
            conf_matrix = confusion_matrix(ytest, class_label_)
            print "confusion matrix \n",conf_matrix
            print "recall =", recall_score(ytest, class_label_, pos_label=churnLabel)
            print "precision =",precision_score(ytest, class_label_, pos_label=churnLabel)
            rocauc = auc(fpr, tpr)
            auc_list.append(rocauc)
            plt.plot(fpr, tpr, color='blue', alpha=.3) #, label='ROC curve (area = %0.2f)' % roc_auc
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.title('Receiver operating characteristics of ' + str(ntrials)+ ' trials')
        pylab.savefig('../images/chapter5_Roc_curve.png')
        plt.close()
        print 'Mean AUC = ', np.mean(auc_list), 'AUC std = ', np.std(auc_list)

        plt.figure(figsize=(18,6));
        plt.hist(active_scores, bins=20, alpha=.3, normed=True, label=activeLabel, color='green');
        plt.hist(churn_scores, bins=20, alpha=.3, normed=True, label=churnLabel, color='red');
        plt.legend()
        plt.title('Score Distribution')
        pylab.savefig('../images/chapter5_score_dist.png')
        plt.close()
        d1 = pd.DataFrame({'score':active_scores, 'Stage':activeLabel})
        d2 = pd.DataFrame({'score':churn_scores, 'Stage':churnLabel})
        return (pd.concat([d1, d2], axis = 0), Xtest, ytest, probas_[:,0], clf)

    @staticmethod
    def build_final_model(features_final, modelFile,STATUS_NAME):
        """ Build the final model
        Parameters:
        -----------
        features_final is the feature/target matrix
        modelFile is the name of the file of the pickled model
        """

        features_final = features_final.dropna(how='any')
        X,y = Chapter5.featuresFinalToXY(features_final,STATUS_NAME)
        clf = RandomForestClassifier(n_estimators=1000, class_weight='auto', n_jobs=-1, oob_score=True, max_depth=3, verbose = 1)
        clf.fit(X,y)
        print clf.oob_score_
        import cPickle
        with open(modelFile, 'wb') as f:
            cPickle.dump(clf, f)
        return clf

    @staticmethod
    def featuresFinalToXY(features_final,STATUS_NAME):
        """ This is a simple utility function that transforms the features,target dataframe into X and Y to be used by
        sklearn """
        features_final = features_final.copy()
        y = features_final.pop(STATUS_NAME).values
        X = features_final.values
        return X,y

    @staticmethod
    def featureImportance(clf, feature_names):
        """ Plot the global feature Importance
        Parameters:
        -----------
        clf: The RandomForest Model
        feature_names: The list of features in the same order as used in the model
        We will not do any error checking here, so we need to be careful when entering the feature_names
        """
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        index = [feature_names[indices[f]] for f in range(len(feature_names))] # rearrange the feature names in the order of importance
        data = [importances[indices[f]] for f in range(len(feature_names))] # The ordered importances
        imp = pd.DataFrame(index = index, data = data, columns = ['Relative Importance']).head(20)
        imp.plot(kind='barh', use_index = True)
        pylab.savefig('../images/chapter5_feature_importance.png')
        plt.close()

    @staticmethod
    def get_top_n_features(clf,feature_names,top_n=1):
        """ Plot the global feature Importance
        Parameters:
        -----------
        clf: The RandomForest Model
        feature_names: The list of features in the same order as used in the model
        top_n: the number of top
        We will not do any error checking here, so we need to be careful when entering the feature_names
        """
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        index = [feature_names[indices[f]] for f in range(len(feature_names))] # rearrange the feature names in the order of importance
        data = [importances[indices[f]] for f in range(len(feature_names))] # The ordered importances
        imp = pd.DataFrame(index = index, data = data, columns = ['Relative Importance']).head(top_n)
        return imp

    @staticmethod
    def build_model_succinct(featureDF, targetVariable, churnLabel, activeLabel):
        """ Build and test models using many trials with random training and test sets
        succint version of the function build_model, is less verbose and smaller no. of iterations
        Parameters:
        -----------
        featureDF: The feature data frame, has one column for the target variable
        targetVariable: The name of the column containing the target variable
        churnLabel: The label of churn in the target variable (e.g. churn or 1)
        activeLabel: The label of active accounts in the target variable (e.g. active or 0)

        Returns:
        --------
        testScoresDF: a data frame containing the scores on different test sets, and the target variable
        Xtest: just the features of the last test set
        ytest: the target variables of the last test set
        scoreTest: the scores of the last test set.
        clf: the last model computed
        The Xtest,ytest and scoreTest will be used to inspect the results to see if they make sense
        clf will be used to check few things in the model (like importance)
        """

        # Since this is a generic ffuction, the safest thing to do is to exclude rows with NAs
        # But we should really deal with NAs before calling this function, because we know how the
        # data was generated.
        # I will assume for now that it is a two class problem, so only 2 classes
        x = featureDF.dropna(how='any')
        y = x.pop(targetVariable) # now x will not contain the target variable
        # What is the distribution of classes?
        print y.value_counts()
        # Now build the models
        ntrials=20
        plt.figure(figsize=(10,10))
        auc_list = list()
        active_scores, churn_scores = list(), list()
        clf = RandomForestClassifier(n_estimators=1000, max_depth= 3,
                                         oob_score=True, n_jobs=-1,
                                         class_weight='auto') #shallow trees, weights inversely proportional to class freqs
        for i in range(ntrials):
            Xtrain, Xtest, ytrain, ytest = train_test_split(x, y)
            clf.fit(Xtrain, ytrain)
            probas_ = clf.predict_proba(Xtest)
            class_label_ = clf.predict(Xtest)

            churn_scores.extend(probas_[np.where(ytest==churnLabel)[0],0])
            active_scores.extend(probas_[np.where(ytest==activeLabel)[0],0]);

            fpr, tpr, thresholds = roc_curve(ytest, probas_[:, 1], pos_label=churnLabel)
            conf_matrix = confusion_matrix(ytest, class_label_)
            rocauc = auc(fpr, tpr)
            auc_list.append(rocauc)
            plt.plot(fpr, tpr, color='blue', alpha=.3) #, label='ROC curve (area = %0.2f)' % roc_auc
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.title('Receiver operating characteristics of ' + str(ntrials)+ ' trials')
        plt.show()
        print 'Mean AUC = ', np.mean(auc_list), 'AUC std = ', np.std(auc_list)

        plt.figure(figsize=(18,6));
        plt.hist(active_scores, bins=20, alpha=.3, normed=True, label=activeLabel, color='green');
        plt.hist(churn_scores, bins=20, alpha=.3, normed=True, label=churnLabel, color='red');
        plt.legend()
        plt.title('Score Distribution')
        d1 = pd.DataFrame({'score':active_scores, 'Stage':activeLabel})
        d2 = pd.DataFrame({'score':churn_scores, 'Stage':churnLabel})
        return (pd.concat([d1, d2], axis = 0), Xtest, ytest, probas_[:,0], clf)

if __name__ == '__main__':

    account_col = 'account_id'
    week_col = 'week'
    status_label = 'status'
    status_active = 'active'
    status_churn = 'churn'

    features_final = pd.read_csv("../data/features_final.csv")
    features_final.set_index(account_col, inplace=True)

    testScoresDF, Xtest, ytest, scores, clf = Chapter5.build_model(features_final, targetVariable=status_label, churnLabel=status_churn, activeLabel=status_active)
    Chapter5.build_final_model(features_final, "../models/marin.clf",status_label)
    feature_names = list(features_final.columns.values)
    feature_names.remove(status_label)
    Chapter5.featureImportance(clf, feature_names)

    # feature_finals_bkp = features_final
    # for iteration in xrange(1,10):
    #     print "iteration",iteration
    #     feature_names = list(features_final.columns.values)
    #     feature_names.remove(status_label)
    #     features_final.drop(Chapter5.get_top_n_features(clf,feature_names,2).index,axis=1,inplace=True)
    #     print "features dropped", Chapter5.get_top_n_features(clf,feature_names,2).index
    # #     print "features used", features_final.columns.values
    #     testScoresDF, Xtest, ytest, scores, clf = Chapter5.build_model_succinct(features_final, targetVariable=Status_Label, churnLabel=CHURN, activeLabel=ACTIVE)
    #     feature_names = list(features_final.columns.values)
    #     feature_names.remove(status_label)
    #     Chapter5.featureImportance(clf, feature_names)

