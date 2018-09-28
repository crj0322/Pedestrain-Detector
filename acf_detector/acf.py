import cv2
import numpy as np
import os
from skimage.io import imread
from sklearn.externals import joblib
from tqdm import tqdm
import re
import time


def rgb2acf(img, pool_size=4):
    
    # smooth
    smooth_kernel = np.array([[1., 2., 1.]])/4.
    img_smooth = cv2.filter2D(img, cv2.CV_32F, smooth_kernel)
    
    # calc gradient
    gradient_kernel = np.array([[-1., 0., 1.]])
    img_dx = cv2.filter2D(img_smooth, cv2.CV_32F, gradient_kernel)
    img_dy = cv2.filter2D(img_smooth, cv2.CV_32F, gradient_kernel.T)
    img_mag = cv2.magnitude(img_dx, img_dy)
    max_channel = np.argmax(img_mag, axis=-1)
    pos = (np.arange(img.shape[0])[:,None], np.arange(img.shape[1]), max_channel)
    img_dx = img_dx[pos]
    img_dy = img_dy[pos]
    img_grad = img_mag[pos]
    
    # normalized gradient magnitude
    exp_grad = cv2.boxFilter(img_grad, cv2.CV_32F, (11, 11)) # to be replaced by triangle filter
    norm_grad = img_grad/(exp_grad + 0.005)
    
    # gradient angle
    img_dx = np.where((img_dx >= 0) & (img_dx < 0.1), 0.1, img_dx)
    img_dx = np.where((img_dx >= -0.1) & (img_dx < 0), -0.1, img_dx)
    grad_angle = np.arctan(img_dy/img_dx)
    grad_angle[(img_dx < 0) & (img_dy > 0)] += np.pi
    grad_angle[(img_dx < 0) & (img_dy < 0)] -= np.pi
    
    # hog channels of 6 bins
    hog_channels = np.zeros((img.shape[0], img.shape[1], 6), dtype=np.float32)
    for i in range(6):
        angle_start = i*np.pi/3. - np.pi
        angle_end = angle_start + np.pi/3.
        hog_channels[:,:,i] = np.where(((grad_angle > angle_start) & (grad_angle <= angle_end)), norm_grad, 0)
    
    # rgb to luv
    img_smooth /= 255.
    img_luv = cv2.cvtColor(img_smooth, cv2.COLOR_RGB2LUV)
    
    # normalize luv
    img_luv[:,:,1] += 88.
    img_luv[:,:,2] += 134.
    img_luv /= 270.
    
    # concatenate channels
    img_channels = np.concatenate((norm_grad[:,:,None], hog_channels, img_luv), axis=-1)

    # 4*4 sum pooling
    pooling_shape = (img_channels.shape[0]//pool_size, img_channels.shape[1]//pool_size, img_channels.shape[2])
    img_pooling = np.zeros(pooling_shape, dtype=np.float32)
    for i in range(pooling_shape[0]):
        for j in range(pooling_shape[1]):
            img_pooling[i, j, :] = np.sum(img_channels[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size, :], axis=(0, 1))
    
    # smooth
    smooth_kernel = np.array([[1., 2., 1.]])/4.
    features = cv2.filter2D(img_pooling, cv2.CV_32F, smooth_kernel)
    
    return features


def random_crop(image, crop_shape):
    org_shape = image.shape
    nh = np.random.randint(0, org_shape[0] - crop_shape[0])
    nw = np.random.randint(0, org_shape[1] - crop_shape[1])
    image_crop = image[nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
    
    return image_crop


def random_gen(file_dir, data_shape):
    # generate negetave data
    file_list = os.listdir(file_dir)

    # read all img in RAM for quick access
    img_list = []
    for file_name in file_list:
        file_path = os.path.join(file_dir, file_name)
        img = (imread(file_path)[...,:3]).astype(np.uint8)
        img_list.append(img)
    
    while True:
        file_index = np.random.randint(0, len(file_list))
        img = img_list[file_index]
        crop_img = random_crop(img, data_shape)
        features = rgb2acf(crop_img).flatten()
        yield features


def generate_data(pos_dir, neg_dir, data_shape=(128, 64), neg_sample_size=5000):
    feature_dim = data_shape[0]*data_shape[1]*10//16 # 5120

    # generate positive data
    file_list = os.listdir(pos_dir)
    X_pos = np.zeros((len(file_list), feature_dim), dtype=np.float32)
    for i, file in tqdm(enumerate(file_list)):
        file_path = os.path.join(pos_dir, file)
        img = imread(file_path)
        img = img[:, :, :3]
        if img.shape[0] < data_shape[0] or img.shape[1] < data_shape[1]:
            print("need padding: ", file, img.shape)
            pass
        
        if img.shape[0] > data_shape[0]:
            offset = (img.shape[0] - data_shape[0])//2
            img = img[offset:offset+data_shape[0], :, :]
        if img.shape[1] > data_shape[1]:
            offset = (img.shape[1] - data_shape[1])//2
            img = img[:, offset:offset+data_shape[1], :]
        
        X_pos[i] = rgb2acf(img).flatten()

    # generate negetave data
    file_gen = random_gen(neg_dir, data_shape)
    X_neg = np.zeros((neg_sample_size, feature_dim), dtype=np.float32)
    for i in tqdm(range(neg_sample_size)):
        X_neg[i] = next(file_gen)

    # generate labels
    y_pos = np.ones(X_pos.shape[0])
    y_neg = np.zeros(X_neg.shape[0])
    
    # concatenate data
    X = np.concatenate((X_pos, X_neg), axis=0)
    y = np.concatenate((y_pos, y_neg))

    return X, y


def generate_hard_neg(file_dir, clf, data_shape=(128, 64), sample_size=5000, batch_size=500):
    file_gen = random_gen(file_dir, data_shape)
    feature_dim = data_shape[0]*data_shape[1]*10//16
    X = np.zeros((sample_size, feature_dim))
    
    with tqdm(total=sample_size) as pbar:
        index = 0
        while index < sample_size:
            # batch predict for speed acceleration
            features = np.zeros((batch_size, feature_dim))
            for i in range(batch_size):
                features[i] = next(file_gen)
            pred = clf.predict(features)
            features = features[pred == 1]

            next_index = index + features.shape[0]
            if next_index > sample_size:
                next_index = sample_size
            X[index:next_index] = features[:next_index - index]
            pbar.update(next_index - index)
            index = next_index

    y = np.zeros(sample_size)

    return X, y


def sliding_window(image, window_size, step_size):
    for y in range(0, image.shape[0], step_size[0]):
        for x in range(0, image.shape[1], step_size[1]):
            if y + window_size[0] > image.shape[0] or x + window_size[1] > image.shape[1]:
                continue
            yield (x, y, image[y:y + window_size[0], x:x + window_size[1]])


def detect_bak(img, clf, window_hw=(32, 16), pool_size=4, stride=4):
    # approximation scale
    approx_scale = (1.125, 1., 0.875, 0.75)
    
    bbox_list = []
    prob_list = []
    pyramid_scale = 1.
    while True:
        if pyramid_scale == 1.:
            feature_map = rgb2acf(img, pool_size=pool_size)
        else:
            # (src.cols+1)//2, (src.rows+1)//2
            feature_map = cv2.pyrDown(feature_map)
        
        if feature_map.shape[0] < window_hw[0] or feature_map.shape[1] < window_hw[1]:
            break
        
        for s in approx_scale:
            if pyramid_scale*s > 1.:
                continue
            
            cur_shape = (int(feature_map.shape[0]*s), int(feature_map.shape[1]*s))
            if cur_shape[0] < window_hw[0] or cur_shape[1] < window_hw[1]:
                break
                
            approx_map = np.copy(feature_map)

            # for normalized gradient, lambda=0.101
            approx_map[..., :7] *= s**(-0.101)

            # slide
            for (x, y, im_window) in sliding_window(approx_map, window_hw, (stride, stride)):
                features = np.reshape(im_window, (1, -1))
                pred = clf.predict(features)
                prob = clf.decision_function(features)

                # calc bbox size
                if pred:
                    x_min = int(pool_size*x/(pyramid_scale*s))
                    y_min = int(pool_size*y/(pyramid_scale*s))
                    w = int(pool_size*window_hw[1]/(pyramid_scale*s))
                    h = int(pool_size*window_hw[0]/(pyramid_scale*s))
                    bbox_list.append((x_min, y_min, x_min + w, y_min + h))
                    prob_list.append(prob)
        
        pyramid_scale /= 2.
        
    return bbox_list, prob_list


def detect(img, clf, window_hw=(128, 64), pool_size=4, stride=2):
    bbox_list = []
    prob_list = []
    
    # batch predict for speed acceleration
    batch_samples = []

    # element as (x, y, pyramid_count)
    box_info = []

    pyramid_count = 0
    slide_size = (window_hw[0]//pool_size, window_hw[1]//pool_size)
    while True:
        if pyramid_count > 0:
            img = cv2.pyrDown(img)
        
        if img.shape[0] < window_hw[0] or img.shape[1] < window_hw[1]:
            break

        feature_map = rgb2acf(img)
        
        # slide
        for (x, y, im_window) in sliding_window(feature_map, slide_size, (stride, stride)):
            batch_samples.append(im_window.flatten())
            box_info.append(np.array([x, y, pyramid_count]))

        pyramid_count += 1

    batch_samples = np.stack(batch_samples, axis=0)
    box_info = np.stack(box_info, axis=0)

    prob = clf.predict_proba(batch_samples)
    pred = np.argmax(prob, axis=1)

    pred_mask = (pred == 1)
    batch_samples = batch_samples[pred_mask]
    prob = prob[pred_mask]
    box_info = box_info[pred_mask]

    x = box_info[:,0]
    y = box_info[:,1]
    scale = np.power(2, box_info[:,2])

    x_min = (pool_size*x*scale).astype(np.uint16)
    y_min = (pool_size*y*scale).astype(np.uint16)
    w = (window_hw[1]*scale).astype(np.uint16)
    h = (window_hw[0]*scale).astype(np.uint16)

    bbox_list = np.column_stack((x_min, y_min, x_min + w, y_min + h))
    prob_list = np.squeeze(prob[:,1])

    return bbox_list, prob_list


def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

	# if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked
    return boxes[pick].astype("int")


def filter_bbox(bbox_list, prob_list, prob_threshold=0.5, nms_threshold=0.3):
    if prob_threshold > 0:
        bbox_list = bbox_list[prob_list >= prob_threshold]
        prob_list = prob_list[prob_list >= prob_threshold]

    if nms_threshold > 0:
        bbox_list = non_max_suppression(bbox_list, probs=prob_list, overlapThresh=nms_threshold)

    return bbox_list


def draw_bbox(img, bbox_list):
    for(xA, yA, xB, yB) in bbox_list:
        cv2.rectangle(img, (xA, yA), (xB, yB), (255, 0, 0), 2)
    return img


def read_annotation(file_path):
    """
    bbox: (Xmin, Ymin, Xmax, Ymax) 
    """
    bboxes = []
    file = open(file_path, 'r')
    lines = file.readlines()
    for line in lines:
        if line.startswith("Bounding box"):
            box = [int(s) for s in re.findall(r'\d+', line[37:])]
            bboxes.append(box)
    
    return bboxes

def calc_iou(box1, box2):
    xmin_1, ymin_1, xmax_1, ymax_1 = box1
    xmin_2, ymin_2, xmax_2, ymax_2 = box2
    
    w_1 = xmax_1 - xmin_1 + 1
    h_1 = ymax_1 - ymin_1 + 1
    w_2 = xmax_2 - xmin_2 + 1
    h_2 = ymax_2 - ymin_2 + 1
    i_w = min(xmax_1, xmax_2) - max(xmin_1, xmin_2) + 1
    i_h = min(ymax_1, ymax_2) - max(ymin_1, ymin_2) + 1
    intersection = max(i_w, 0) * max(i_h, 0)
    union = w_1 * h_1 + w_2 * h_2 - intersection
    
    iou = float(intersection)/union
    return iou


def main():
    clf = joblib.load(".\\acf_detector\\adaboost.v1.pkl")
    test_dir = "E:\\data\\INRIAPerson\\Test\\pos"
    file_list = os.listdir(test_dir)
    file = os.path.join(test_dir, file_list[114])
    img = imread(file)
    img = img[...,:3]
    """print(img.shape)
    cv2.imshow("detection", img[...,::-1])"""

    start = time.clock()
    bbox_list, prob_list = detect(img, clf)
    rects = filter_bbox(bbox_list, prob_list, prob_threshold=0)
    end = time.clock()
    img_box = draw_bbox(img, rects)
    print("detection spent %.5fs" % (end - start))
    cv2.imshow("detection", img_box[...,::-1])
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()