import h5py
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.metrics import f1_score, accuracy_score


def read_features(file):
    with h5py.File(file, 'r') as h:
        pos_feat = np.array(h['pos_feat'])
        neg_feat = np.array(h['neg_feat'])

    pos_label = np.ones((pos_feat.shape[0],))
    neg_label = np.zeros((neg_feat.shape[0],))
    
    return pos_feat, pos_label, neg_feat, neg_label


def read_hard_neg(file):
    with h5py.File(file, 'r') as h:
        hard_neg = np.array(h['hard_neg_feat'])

    hard_neg_label = np.zeros((hard_neg.shape[0],))
    
    return hard_neg, hard_neg_label


def train_model(model_file, feat_file, hard_neg_file=None):
    if os.path.exists(feat_file):
        pos_feat, pos_label, neg_feat, neg_label = read_features(feat_file)
    else:
        print("Features not exist!")
        return
    
    if hard_neg_file and os.path.exists(hard_neg_file):
        hard_neg, hard_neg_label = read_hard_neg(hard_neg_file)
        X = np.concatenate((pos_feat, neg_feat, hard_neg), axis=0)
        y = np.concatenate((pos_label, neg_label, hard_neg_label), axis=0)
    else:
        X = np.concatenate((pos_feat, neg_feat), axis=0)
        y = np.concatenate((pos_label, neg_label), axis=0)
        
    clf = LinearSVC()
    clf.fit(X, y)
    y_pred = clf.predict(X)
    f1 = f1_score(y, y_pred)
    print('Train f1: ', f1)
    acc = accuracy_score(y, y_pred)
    print('Train acc: ', acc)
    
    joblib.dump(clf, model_file)
    
    
def main():
    data_path = ".." + os.sep + "data" + os.sep
    if os.path.exists(data_path + "hard_neg.h5"):
        train_model(data_path + "svc_v2.pkl", data_path + "features.h5", data_path + "hard_neg.h5")
    else:
        train_model(data_path + "svc_v1.pkl", data_path + "features.h5")

        
if __name__ == "__main__":
    main()