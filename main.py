import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import tree, svm
import sys


def loadData(file_path):

    df_data = pd.read_csv(file_path, sep=',', header=None)
    spam = df_data.loc[df_data[57] == 1]
    non_spam = df_data.loc[df_data[57] == 0]
    non_spam = non_spam.sample(frac=1)
    non_spam = non_spam[0:len(spam)]

    df_data = pd.concat([spam, non_spam])
    df_data = df_data.sample(frac=1)
    size = len(df_data)
    divider = int(size*0.8)

    train_data = df_data[:divider]
    test_data = df_data[divider:]

    return train_data, test_data


def train(train_data, clf):

    spam_indicator = np.array(train_data.iloc[:,-1])
    attributes = np.array(train_data.iloc[:,:-1])
    clf.fit(attributes, spam_indicator)

    return clf


def test(test_data, clf):

    spam_indicator = np.array(test_data.iloc[:,-1])
    attributes = np.array(test_data.iloc[:,:-1])
    pred = clf.predict(attributes)
    size = len(pred)
    counter = 0

    for i in range(0, size):
        counter += 1 if pred[i] != spam_indicator[i] else 0
    mislabeled = 100 - counter*100/size

    return round(mislabeled, 2)


def main():

    file_path = "spambase.data"
    train_data, test_data = loadData(file_path)

    gnb_clf = train(train_data, GaussianNB())
    dtc_clf = train(train_data, tree.DecisionTreeClassifier())
    svm_clf = train(train_data, svm.SVC())

    gnb_ = test(test_data, gnb_clf)
    dtc_ = test(test_data, dtc_clf)
    svm_ = test(test_data, svm_clf)

    print('Naive-Bayes: {}'.format(gnb_))
    print('Decision Tree: {}'.format(dtc_))
    print('SVM: {}'.format(svm_))


if __name__ == '__main__':
    main()
