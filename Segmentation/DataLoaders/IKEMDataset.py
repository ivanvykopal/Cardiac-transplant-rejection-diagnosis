from tensorflow.keras.utils import Sequence
import numpy as np

from Utils.Preprocessing.MaskCreator import MaskCreator
from Utils.Preprocessing.Reader import Reader
from Utils.augmentations import augmentate


class Dataset(Sequence):
    def __init__(self, batch_size, directory, img_json, config, augmentations):
        self.mask_creator = MaskCreator('files')
        self.reader = Reader(
            self.mask_creator,
            directory=f'{directory}/images'
        )

        self.batch_size = batch_size
        self.img_json = img_json
        self.config = config
        self.img_paths = []

        self.annotation_classes = self.config['classes']
        self.augmentations = [{"type": "normal"}] + augmentations
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

        x = np.full(
            (self.batch_size,) + (self.config['image_size'], self.config['image_size']) + (self.config['channels'],),
            dtype="float32",
            fill_value=0
        )

        y = np.zeros(
            (self.batch_size,) + (self.config['image_size'], self.config['image_size']) + (len(self.config['classes']),),
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

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.img_paths))
        np.random.shuffle(self.indexes)
