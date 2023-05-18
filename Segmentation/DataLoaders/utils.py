import cv2 as cv
import numpy as np


def convert_to_multiclass(mask, config):
    union = mask[:, :, 1]
    for i in range(2, len(config['final_classes'])):
        union = cv.bitwise_or(union, mask[:, :, i])
    union = np.clip(union, 0, 1)
    mask[:, :, 0] = np.where((union == 0) | (union == 1), union ^ 1, union)

    return mask
