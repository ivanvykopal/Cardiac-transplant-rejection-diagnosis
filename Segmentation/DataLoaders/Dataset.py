from tensorflow.keras.utils import Sequence
import numpy as np

from Utils.Preprocessing.MaskCreator import MaskCreator
from Utils.Preprocessing.Reader import Reader
from Utils.augmentations import augmentate


class Dataset(Sequence):
    def __init__(self, batch_size, directory, img_json, config, in_num, out_num, data_type='files', inference=False):
        if data_type == 'files':
            self.mask_creator = MaskCreator('files')
        else:
            self.mask_creator = MaskCreator('hasura', 'http://localhost:8080/v1/graphql', None)

        self.reader = Reader(
            self.mask_creator,
            directory=f'{directory}',
            inference=self.inference
        )

        self.batch_size = batch_size
        self.img_json = img_json
        self.config = config
        self.in_num = in_num
        self.out_num = out_num
        self.img_paths = []
        self.inference = inference

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

        for index, image in enumerate(self.img_json):
            coors = image['patch'] \
                    + [image['augmentation'], {
                        'name': image['name'].replace('.tif', '').replace('.png', ''),
                        'height': image['height'],
                        'width': image['width']
                    }, {
                           'idx': image['fragment']['idx'],
                           'x': image['fragment']['x'],
                           'y': image['fragment']['y'],
                           'width': image['fragment']['width'],
                           'height': image['fragment']['height']
                       }]
            self.img_paths.append(coors)

        self.on_epoch_end()

    def __len__(self):
        return len(self.img_paths) // self.batch_size

    def __getitem__(self, idx):
        i = idx * self.batch_size
        indexes = self.indexes[i: i + self.batch_size]
        batch_imgs = [self.img_paths[k] for k in indexes]

        return self.get_data(batch_imgs)

    def _in1_out1(self, batch_imgs):
        x = np.full(
            (self.batch_size,) + (self.config['image_size'], self.config['image_size']) + (self.config['channels'],),
            dtype="float32",
            fill_value=1
        )

        y = np.zeros(
            (self.batch_size,) + (self.config['image_size'], self.config['image_size']) + (self.classes,),
            dtype="float32"
        )

        for index, batch in enumerate(batch_imgs):
            y_idx, x_idx, augmentation, image, fragment = batch

            img, mask = self.reader.read_patch(
                image=image,
                fragment=fragment,
                x=x_idx,
                y=y_idx,
                patch_size=self.config['image_size'],
                annotation_classes=self.annotation_classes
            )

            x[index], y[index] = augmentate(img=img, mask=mask, augmentation=augmentation)

        return x, y

    def _in2_out1(self, batch_imgs):
        x1 = np.full(
            (self.batch_size,) + (self.config['image_size'], self.config['image_size']) + (self.config['channels'],),
            dtype="float32", fill_value=1
        )

        x2 = np.full(
            (self.batch_size,) + (self.config['image_size'] // 2, self.config['image_size'] // 2) + (3,),
            dtype="float32", fill_value=1
        )

        y = np.zeros(
            (self.batch_size,) + (self.config['image_size'], self.config['image_size']) + (self.classes,),
            dtype="float32"
        )

        for index, batch in enumerate(batch_imgs):
            y_idx, x_idx, augmentation, image, fragment = batch

            img_scale0, img_scale1, mask = self.reader.read_in2_out1(
                image=image,
                fragment=fragment,
                x=x_idx,
                y=y_idx,
                patch_size=self.config['image_size'],
                annotation_classes=self.annotation_classes,
                use_cells=(self.config['channels'] == 4)
            )

            x1[index], y[index] = augmentate(img=img_scale0, mask=mask, augmentation=augmentation)
            x2[index], y[index] = augmentate(img=img_scale1, mask=mask, augmentation=augmentation)

        return [x1, x2], y

    def _in3_out1(self, batch_imgs):
        x1 = np.full(
            (self.batch_size,) + (self.config['image_size'], self.config['image_size']) + (self.config['channels'],),
            dtype="float32",
            fill_value=1
        )
        x2 = np.full(
            (self.batch_size,) + (self.config['image_size'], self.config['image_size']) + (3,),
            dtype="float32",
            fill_value=1
        )
        x3 = np.full(
            (self.batch_size,) + (self.config['image_size'], self.config['image_size']) + (3,),
            dtype="float32",
            fill_value=1
        )

        y = np.zeros(
            (self.batch_size,) + (self.config['image_size'], self.config['image_size']) + (self.classes,),
            dtype="float32"
        )

        for index, batch in enumerate(batch_imgs):
            y_idx, x_idx, augmentation, image, fragment = batch

            img_scale0, img_scale1, img_scale2, mask = self.reader.read_in3_out1(
                image=image,
                fragment=fragment,
                x=x_idx,
                y=y_idx,
                patch_size=self.config['image_size'],
                annotation_classes=self.annotation_classes,
                use_cells=(self.config['channels'] == 4)
            )

            x1[index], y[index] = augmentate(img=img_scale0, mask=mask, augmentation=augmentation)
            x2[index], y[index] = augmentate(img=img_scale1, mask=mask, augmentation=augmentation)
            x3[index], y[index] = augmentate(img=img_scale2, mask=mask, augmentation=augmentation)

        return [x1, x2, x3], y

    def _in2_out3(self, batch_imgs):
        x1 = np.full(
            (self.batch_size,) + (self.config['image_size'], self.config['image_size']) + (self.config['channels'],),
            dtype="float32", fill_value=1)
        x2 = np.full((self.batch_size,) + (self.config['image_size'] * 2, self.config['image_size'] * 2) + (
            self.config['channels'],), dtype="float32", fill_value=1)

        y1 = np.zeros((self.batch_size,) + (self.config['image_size'], self.config['image_size']) + (self.classes,),
                      dtype="float32")
        y2 = np.zeros(
            (self.batch_size,) + (self.config['image_size'] * 2, self.config['image_size'] * 2) + (self.classes,),
            dtype="float32")

        for index, batch in enumerate(batch_imgs):
            y_idx, x_idx, augmentation, image, fragment = batch

            img1, img2, mask1, mask2 = self.reader.read_in2_out2(
                image=image,
                fragment=fragment,
                x=x_idx,
                y=y_idx,
                patch_size=self.config['image_size'],
                annotation_classes=self.annotation_classes,
                use_cells=(self.config['channels'] == 4),
                lab=self.config['lab']
            )

            x1[index], y1[index] = augmentate(img=img1, mask=mask1, augmentation=augmentation)
            x2[index], y2[index] = augmentate(img=img2, mask=mask2, augmentation=augmentation)

        combined = np.concatenate([y1, y1], axis=-1)
        return [x1, x2], [y1, y2, combined]

    def _in3_out4(self, batch_imgs):
        x1 = np.full(
            (self.batch_size,) + (self.config['image_size'], self.config['image_size']) + (self.config['channels'],),
            dtype="float32", fill_value=1)
        x2 = np.full((self.batch_size,) + (self.config['image_size'] * 2, self.config['image_size'] * 2) + (
            self.config['channels'],), dtype="float32", fill_value=1)
        x3 = np.full((self.batch_size,) + (self.config['image_size'] * 4, self.config['image_size'] * 4) + (
            self.config['channels'],), dtype="float32", fill_value=1)

        y1 = np.zeros((self.batch_size,) + (self.config['image_size'], self.config['image_size']) + (self.classes,),
                      dtype="float32")
        y2 = np.zeros(
            (self.batch_size,) + (self.config['image_size'] * 2, self.config['image_size'] * 2) + (self.classes,),
            dtype="float32")
        y3 = np.zeros(
            (self.batch_size,) + (self.config['image_size'] * 4, self.config['image_size'] * 4) + (self.classes,),
            dtype="float32")

        for index, batch in enumerate(batch_imgs):
            y_idx, x_idx, augmentation, image, fragment = batch

            img1, img2, img3, mask1, mask2, mask3 = self.reader.read_in3_out3(
                image=image,
                fragment=fragment,
                x=x_idx,
                y=y_idx,
                patch_size=self.config['image_size'],
                annotation_classes=self.annotation_classes,
                use_cells=(self.config['channels'] == 4)
            )

            x1[index], y1[index] = augmentate(img=img1, mask=mask1, augmentation=augmentation)
            x2[index], y2[index] = augmentate(img=img2, mask=mask2, augmentation=augmentation)
            x3[index], y3[index] = augmentate(img=img3, mask=mask3, augmentation=augmentation)

        combined = np.concatenate([y1, y1, y1], axis=-1)
        return [x1, x2, x3], [y1, y2, y3, combined]

    def _in1_out3(self, batch_imgs):
        x = np.full(
            (self.batch_size,) + (self.config['image_size'], self.config['image_size']) + (self.config['channels'],),
            dtype="float32", fill_value=1)

        y1 = np.zeros(
            (self.batch_size,) + (self.config['image_size'], self.config['image_size']) + (1,),
            dtype="float32"
        )
        y2 = np.zeros(
            (self.batch_size,) + (self.config['image_size'], self.config['image_size']) + (1,),
            dtype="float32"
        )
        y3 = np.zeros(
            (self.batch_size,) + (self.config['image_size'], self.config['image_size']) + (1,),
            dtype="float32"
        )

        for index, batch in enumerate(batch_imgs):
            y_idx, x_idx, augmentation, image, fragment = batch

            img, mask1, mask2, mask3 = self.reader.read_in1_out3(
                image=image,
                fragment=fragment,
                x=x_idx,
                y=y_idx,
                patch_size=self.config['image_size'],
                annotation_classes=self.annotation_classes,
                use_cells=(self.config['channels'] == 4),
                lab=self.config['lab']
            )

            x[index], y1[index] = augmentate(img=img, mask=mask1, augmentation=augmentation)
            _, y2[index] = augmentate(img=img, mask=mask2, augmentation=augmentation)
            _, y3[index] = augmentate(img=img, mask=mask3, augmentation=augmentation)

        return x, [y1, y2, y3]

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.img_paths))
        if not self.inference:
            np.random.shuffle(self.indexes)
