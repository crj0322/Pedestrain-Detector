import random


def random_crop(image, crop_shape):
    org_shape = image.shape
    nh = random.randint(0, org_shape[0] - crop_shape[0])
    nw = random.randint(0, org_shape[1] - crop_shape[1])
    image_crop = image[nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
    
    return image_crop


def sliding_window(image, window_size, step_size):
    for h in range(0, image.shape[0], step_size[0]):
        for w in range(0, image.shape[1], step_size[1]):
            yield (w, h, image[h:h + window_size[0], w:w + window_size[1]])

