import numpy as np


def create_prediction_img(true, pred):
    new_image = np.zeros((pred.shape[0], pred.shape[1], 3))
    new_image[:, :, 0] = true
    new_image[:, :, 1] = pred

    return new_image


def create_RGB_mask(mask, labels, size):
    canvas = np.zeros((size, size, 3))
    for x in range(size):
        for y in range(size):
            canvas[x, y, :] = labels[mask[x, y]]

    return canvas


def get_labels():
    return np.asarray(
        [
            [0, 0, 0], # background
            [128, 0, 0], #blood_vessels
            [0, 128, 0], # inflammations
            [128, 128, 0], # endocard
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [0, 255, 192],
            [255, 170, 0],
            [192, 128, 255]
        ]
    )
