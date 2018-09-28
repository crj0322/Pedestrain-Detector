from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.externals import joblib
import numpy as np
import os
import time

from acf import generate_data


def main():
    # prepare train data
    X_train = joblib.load(".\\acf_detector\\X_train.pkl")
    y_train = joblib.load(".\\acf_detector\\y_train.pkl")
    X_hn = joblib.load(".\\acf_detector\\X_hn.pkl")
    y_hn = np.zeros(X_hn.shape[0])
    X_train = np.concatenate((X_train, X_hn), axis=0)
    y_train = np.concatenate((y_train, y_hn), axis=0)
    X_train, y_train = shuffle(X_train, y_train)

    # train model
    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=2048, learning_rate=0.01)
    clf.fit(X_train, y_train)
    joblib.dump(clf, "adaboost.v2.1.pkl")

    # clf = joblib.load(".\\acf_detector\\adaboost.v2.pkl")

    # prepare test data
    pos_dir = os.path.join("E:/", "data", "INRIAPerson", "test_64x128_H96", "pos")
    neg_dir = os.path.join("E:/", "data", "INRIAPerson", "Test", "neg")
    X_test, y_test = generate_data(pos_dir, neg_dir)

    # evaluate performance
    train_acc = clf.score(X_train, y_train)
    print("Train accuracy: %.5f" % train_acc)

    test_acc = clf.score(X_test, y_test)
    print("Test accuracy: %.5f" % test_acc)

    # evaluate speed
    start = time.clock()
    _ = clf.predict_proba(X_test)
    end = time.clock()
    print('Predict spent time: {0:.5f}s'.format((end-start)/X_test.shape[0]))


if __name__ == "__main__":
    main()