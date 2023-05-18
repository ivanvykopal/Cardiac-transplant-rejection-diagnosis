import cv2 as cv
import numpy as np
from tensorflow.keras.utils import Sequence
from .utils import normalize_lab


class Dataset(Sequence):
    def __init__(self, images, config):
        self.batch_size = config['batch_size']
        self.data = images
        self.config = config
        self.in_num = config['input_images']
        self.out_num = config['output_masks']
        self.lab = config['lab']

        if self.out_num == 1:
            if self.in_num == 1:
                self.get_data = self._in1_out1
            elif self.in_num == 2:
                self.get_data = self._in2_out1
            elif self.in_num == 3:
                self.get_data = self._in3_out1
        elif self.out_num == 3 and self.in_num == 2:
            self.get_data = self._in2_out3
        elif self.out_num == 3 and self.in_num == 1:
            self.get_data = self._in1_out3
        elif self.out_num == 4 and self.in_num == 3:
            self.get_data = self._in3_out4
        else:
            raise ValueError

        self.annotation_classes = self.config['classes']
        self.classes = len(self.annotation_classes)
        self.indexes = np.arange(len(self.data))

    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self, idx):
        i = idx * self.batch_size
        indexes = self.indexes[i: i + self.batch_size]
        batch_imgs = [self.data[k] for k in indexes]

        return self.get_data(batch_imgs)

    def _in1_out1(self, batch_imgs):
        x = np.full(
            (self.batch_size,) + (self.config['image_size'],
                                  self.config['image_size']) + (self.config['channels'],),
            dtype="float32",
            fill_value=1.
        )

        y = np.zeros(
            (self.batch_size,) +
            (self.config['image_size'],
             self.config['image_size']) + (self.classes,),
            dtype="float32"
        )

        for index, batch in enumerate(batch_imgs):
            img = batch[0]

            mask = np.zeros(
                (self.config['image_size'], self.config['image_size'], self.classes))

            if self.lab:
                if self.config['channels'] == 4:
                    img_patch, cell_patch = img[:, :, :3], img[:, :, 3]
                    img_patch = cv.cvtColor(
                        img_patch.astype('float32'), cv.COLOR_RGB2Lab)
                    img_patch = normalize_lab(img_patch)
                    img = np.concatenate(
                        (img_patch, np.expand_dims(cell_patch, -1)),
                        axis=-1
                    )
                else:
                    img = cv.cvtColor(img.astype('float32'), cv.COLOR_RGB2Lab)
                    img = normalize_lab(img)

            x[index], y[index] = img, mask

        return x, y

    def _in2_out1(self, batch_imgs):
        x1 = np.full(
            (self.batch_size,) + (self.config['image_size'],
                                  self.config['image_size']) + (self.config['channels'],),
            dtype="float32", fill_value=1.
        )

        x2 = np.full(
            (self.batch_size,) +
            (self.config['image_size'] // 2,
             self.config['image_size'] // 2) + (3,),
            dtype="float32", fill_value=1.
        )

        y = np.zeros(
            (self.batch_size,) +
            (self.config['image_size'],
             self.config['image_size']) + (self.classes,),
            dtype="float32"
        )

        for index, batch in enumerate(batch_imgs):
            img_scale0, img_scale1 = batch

            mask = np.zeros(
                (self.config['image_size'], self.config['image_size'], self.classes))

            x1[index], y[index] = img_scale0, mask
            x2[index] = img_scale1

        return [x1, x2], y

    def _in3_out1(self, batch_imgs):
        x1 = np.full(
            (self.batch_size,) + (self.config['image_size'],
                                  self.config['image_size']) + (self.config['channels'],),
            dtype="float32",
            fill_value=1.
        )
        x2 = np.full(
            (self.batch_size,) +
            (self.config['image_size'], self.config['image_size']) + (3,),
            dtype="float32",
            fill_value=1.
        )
        x3 = np.full(
            (self.batch_size,) +
            (self.config['image_size'], self.config['image_size']) + (3,),
            dtype="float32",
            fill_value=1.
        )

        y = np.zeros(
            (self.batch_size,) +
            (self.config['image_size'],
             self.config['image_size']) + (self.classes,),
            dtype="float32"
        )

        for index, batch in enumerate(batch_imgs):
            img_scale0, img_scale1, img_scale2 = batch

            mask = np.zeros(
                (self.config['image_size'], self.config['image_size'], self.classes))

            x1[index], y[index] = img_scale0, mask
            x2[index] = img_scale1
            x3[index] = img_scale2

        return [x1, x2, x3], y

    def _in2_out3(self, batch_imgs):
        x1 = np.full(
            (self.batch_size,) + (self.config['image_size'],
                                  self.config['image_size']) + (self.config['channels'],),
            dtype="float32", fill_value=1.)
        x2 = np.full((self.batch_size,) + (self.config['image_size'] * 2, self.config['image_size'] * 2) + (
            self.config['channels'],), dtype="float32", fill_value=1.)

        y1 = np.zeros((self.batch_size,) + (self.config['image_size'], self.config['image_size']) + (self.classes,),
                      dtype="float32")
        y2 = np.zeros(
            (self.batch_size,) + (self.config['image_size'] * 2,
                                  self.config['image_size'] * 2) + (self.classes,),
            dtype="float32")

        for index, batch in enumerate(batch_imgs):
            img1, img2 = batch

            mask1 = np.zeros(
                (self.config['image_size'], self.config['image_size'], self.classes))

            mask2 = np.zeros(
                (self.config['image_size'] * 2, self.config['image_size'] * 2, self.classes))

            x1[index], y1[index] = img1, mask1
            x2[index], y2[index] = img2, mask2

        combined = np.concatenate([y1, y1], axis=-1)
        return [x1, x2], [y1, y2, combined]

    def _in3_out4(self, batch_imgs):
        x1 = np.full(
            (self.batch_size,) + (self.config['image_size'],
                                  self.config['image_size']) + (self.config['channels'],),
            dtype="float32", fill_value=1.)
        x2 = np.full((self.batch_size,) + (self.config['image_size'] * 2, self.config['image_size'] * 2) + (
            self.config['channels'],), dtype="float32", fill_value=1.)
        x3 = np.full((self.batch_size,) + (self.config['image_size'] * 4, self.config['image_size'] * 4) + (
            self.config['channels'],), dtype="float32", fill_value=1.)

        y1 = np.zeros((self.batch_size,) + (self.config['image_size'], self.config['image_size']) + (self.classes,),
                      dtype="float32")
        y2 = np.zeros(
            (self.batch_size,) + (self.config['image_size'] * 2,
                                  self.config['image_size'] * 2) + (self.classes,),
            dtype="float32")
        y3 = np.zeros(
            (self.batch_size,) + (self.config['image_size'] * 4,
                                  self.config['image_size'] * 4) + (self.classes,),
            dtype="float32")

        for index, batch in enumerate(batch_imgs):
            img1, img2, img3 = batch

            mask1 = np.zeros(
                (self.config['image_size'], self.config['image_size'], self.classes))

            mask2 = np.zeros(
                (self.config['image_size'] * 2, self.config['image_size'] * 2, self.classes))

            mask3 = np.zeros(
                (self.config['image_size'] * 4, self.config['image_size'] * 4, self.classes))

            x1[index], y1[index] = img1, mask1
            x2[index], y2[index] = img2, mask2
            x3[index], y3[index] = img3, mask3

        combined = np.concatenate([y1, y1, y1], axis=-1)
        return [x1, x2, x3], [y1, y2, y3, combined]

    def _in1_out3(self, batch_imgs):
        x = np.full(
            (self.batch_size,) + (self.config['image_size'],
                                  self.config['image_size']) + (self.config['channels'],),
            dtype="float32", fill_value=1.)

        y1 = np.zeros(
            (self.batch_size,) +
            (self.config['image_size'], self.config['image_size']) + (1,),
            dtype="float32"
        )
        y2 = np.zeros(
            (self.batch_size,) +
            (self.config['image_size'], self.config['image_size']) + (1,),
            dtype="float32"
        )
        y3 = np.zeros(
            (self.batch_size,) +
            (self.config['image_size'], self.config['image_size']) + (1,),
            dtype="float32"
        )

        for index, batch in enumerate(batch_imgs):
            img = batch[0]

            mask = np.zeros(
                (self.config['image_size'], self.config['image_size'], 1))

            x[index], y1[index] = img, mask
            y2[index] = mask
            y3[index] = mask

        return x, [y1, y2, y3]
