from skimage.feature import hog
from skimage.io import imread
from skimage.transform import pyramid_gaussian
from sklearn.externals import joblib
from tqdm import tqdm
import numpy as np
import h5py
import os
from utils import random_crop, sliding_window


def get_pos_features(data_dir, des_shape=(128, 64), orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), normalize='L2-Hys'):
    file_list = os.listdir(data_dir)
    feature_list = []
    for file in tqdm(file_list):
        img = imread(data_dir + os.sep + file, as_gray=True)
        
        # to des_shape
        if img.shape[0] < des_shape[0] or img.shape[1] < des_shape[1]:
            continue
        
        h_start = (img.shape[0] - des_shape[0])//2
        h_end = h_start + des_shape[0]
        w_start = (img.shape[1] - des_shape[1])//2
        w_end = w_start + des_shape[1]
        img = img[h_start:h_end, w_start:w_end]
        features = hog(img, orientations, pixels_per_cell, cells_per_block, normalize)
        feature_list.append(features.reshape(1, -1))
        
    return np.concatenate(feature_list, axis=0)


def get_neg_features(data_dir, crop_num=10, des_shape=(128, 64), orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), normalize='L2-Hys'):
    file_list = os.listdir(data_dir)
    feature_list = []
    for file in tqdm(file_list):
        img = imread(data_dir + os.sep + file, as_gray=True)
        
        # to des_shape
        if img.shape[0] < des_shape[0] or img.shape[1] < des_shape[1]:
            continue
            
        for i in range(crop_num):
            crop_img = random_crop(img, des_shape)
            features = hog(crop_img, orientations, pixels_per_cell, cells_per_block, normalize)
            feature_list.append(features.reshape(1, -1))
            
    return np.concatenate(feature_list, axis=0)


def save_features(feat_list, feat_name, file_name):
    if os.path.isfile(file_name):
        os.remove(file_name)

    with h5py.File(file_name) as h:
        for feat, name in zip(feat_list, feat_name):
            h.create_dataset(name, data=feat)
        

def gen_hard_features(file, clf, downscale, min_hw=(128, 64), step_size=(10, 10),
                      orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), normalize='L2-Hys'):
    img = imread(file, as_gray=True)
    
    feat_list = []
    for im_scaled in pyramid_gaussian(img, downscale=downscale, multichannel=False):
        if im_scaled.shape[0] < min_hw[0] or im_scaled.shape[1] < min_hw[1]:
            break
        for (x, y, im_window) in sliding_window(im_scaled, min_hw, step_size):
            if im_window.shape[0] != min_hw[0] or im_window.shape[1] != min_hw[1]:
                continue
            
            features = hog(im_window, orientations, pixels_per_cell, cells_per_block, normalize)
            features = np.reshape(features, (1, -1))
            pred = clf.predict(features)
            # dist = clf.decision_function(features)
            
            if pred == 1:
                feat_list.append(features)
        
    return feat_list


def hard_neg_mining(data_dir, clf, downscale):
    file_list = os.listdir(data_dir)
    feat_list = []
    for file in tqdm(file_list):
        feat_list += gen_hard_features(data_dir+os.sep+file, clf, downscale)
        
    return np.concatenate(feat_list, axis=0)


def main():
    clf_file = ".." + os.sep + "data" + os.sep + "svc_v1.pkl"
    if os.path.exists(clf_file):
        neg_path = "E:/data/INRIAPerson/Train/neg"
        clf = joblib.load(clf_file)
        hard_neg_feat = hard_neg_mining(neg_path, clf, 1.25)
        
        save_features([hard_neg_feat], ["hard_neg_feat"], "." + os.sep + "data" + os.sep + "hard_neg.h5")
    else:
        pos_path = "E:/data/INRIAPerson/train_64x128_H96/pos"
        neg_path = "E:/data/INRIAPerson/Train/neg"

        pos_feat = get_pos_features(pos_path)
        neg_feat = get_neg_features(neg_path)

        save_features([pos_feat, neg_feat], ["pos_feat", "neg_feat"], "." + os.sep + "data" + os.sep + "features.h5")


if __name__ == "__main__":
    main()