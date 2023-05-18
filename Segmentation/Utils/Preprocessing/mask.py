import numpy as np
import cv2 as cv


def save_img(path_name, data):
    cv.imwrite(path_name, data * 255)


def read_mask(path_name, name):
    mask = np.load(path_name)
    labels = ['blood_vessels', 'endocariums', 'fatty_tissues', 'inflammations']

    #for i in range(mask.shape[2]):
    save_img('images/' + name + '.png', mask[:, :, 0])


if __name__ == '__main__':
    files = ['1230_21_HE', '2724_21_HE', '2940_21_HE', '3291_21_HE', '4342_21_HE', '7385_21_HE']
    for file in files:
        read_mask('labels/' + file + '.npy', file)
