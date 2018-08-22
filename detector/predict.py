from skimage.transform import pyramid_gaussian
from skimage.io import imread, imsave
from skimage.feature import hog
from imutils.object_detection import non_max_suppression
from sklearn.externals import joblib
import cv2
import os
import numpy as np
from utils import sliding_window
import random

            
def predict(file, clf, downscale, threshold=1, min_hw=(128, 64), step_size=(10, 10),
            orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), normalize='L2-Hys'):
    img = imread(file, as_gray=True)
    
    scale = 0
    bbox_list = []
    for im_scaled in pyramid_gaussian(img, downscale=downscale, multichannel=False):
        if im_scaled.shape[0] < min_hw[0] or im_scaled.shape[1] < min_hw[1]:
            break
        for (x, y, im_window) in sliding_window(im_scaled, min_hw, step_size):
            if im_window.shape[0] != min_hw[0] or im_window.shape[1] != min_hw[1]:
                continue
            
            features = hog(im_window, orientations, pixels_per_cell, cells_per_block, normalize)
            features = np.reshape(features, (1, -1))
            pred = clf.predict(features)
            dist = clf.decision_function(features)
            
            if pred == 1 and dist > threshold:
                bbox = (int(x*(downscale**scale)), int(y*(downscale**scale)), dist,
                     int(min_hw[1]*(downscale**scale)),
                     int(min_hw[0]*(downscale**scale)))
                bbox_list.append(bbox)
                
        scale += 1
        
    return bbox_list


def draw_bbox(file, bbox_list, nms=True, threshold=0.3):
    img = imread(file)
    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in bbox_list])
    sc = [score[0] for (x, y, score, w, h) in bbox_list]
    sc = np.array(sc)
    if nms:
        rects = non_max_suppression(rects, probs = sc, overlapThresh = threshold)
    for(xA, yA, xB, yB) in rects:
        cv2.rectangle(img, (xA, yA), (xB, yB), (0, 255, 0), 2)
    return img


def main():
    test_path = "E:/data/INRIAPerson/Test/pos"
    file_list = os.listdir(test_path)
    clf = joblib.load('../data/svc_v2.pkl')
    file_name = file_list[random.randint(0, len(file_list)-1)]
    file = test_path + os.sep + file_name
    bbox_list = predict(file, clf, 1.25, threshold=0.5)
    img = draw_bbox(file, bbox_list)
    imsave(".." + os.sep + "result" + os.sep + file_name, img)

if __name__ == "__main__":
    main()