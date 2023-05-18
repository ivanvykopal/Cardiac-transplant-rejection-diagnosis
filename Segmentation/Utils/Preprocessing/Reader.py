from tensorflow.keras.utils import load_img
import tifffile
import os
import numpy as np
import cv2 as cv

from .MaskCreator import MaskCreator


def normalize_lab(image):
    image[:, :, 0] /= 100
    image[:, :, 1] += 128
    image[:, :, 1] /= 255
    image[:, :, 2] += 128
    image[:, :, 2] /= 255

    return image


class Reader:
    def __init__(self, mask_creator=None, directory=None, inference=False):
        self.directory = directory
        self.inference = inference
        if mask_creator is None:
            self.mask_creator = MaskCreator(None, None)
        else:
            self.mask_creator = mask_creator

    @staticmethod
    def _save_image(image, name, big=False):
        with tifffile.TiffWriter(name, bigtiff=big) as f:
            f.write(image)

    def _get_fragment(self, image, x, y, patch_size, fill_value=255.):
        height, width, channels = image.shape

        canvas = np.full((patch_size, patch_size, channels),
                         fill_value=fill_value)
        try:
            if x < 0 and y < 0 and x + patch_size > width and y + patch_size > height:  # DONE
                canvas[abs(y):abs(y) + height, abs(x): abs(x) +
                       width, :] = image[0:height, 0:width, :]
            elif x < 0 and y < 0 and x + patch_size > width:  # DONE
                canvas[abs(y):patch_size, abs(x):abs(x) + width,
                       :] = image[0:y + patch_size, 0:width, :]
            elif x < 0 and y < 0 and y + patch_size > height:  # DONE
                canvas[abs(y):abs(y) + height, abs(x):patch_size,
                       :] = image[0:height, 0:x + patch_size, :]
            elif x < 0 and y < 0:  # DONE
                canvas[abs(y):patch_size, abs(x):patch_size,
                       :] = image[0: y + patch_size, 0: x + patch_size, :]
            elif x < 0 and x + patch_size > width and y + patch_size > height:  # DONE
                canvas[0:height - y, abs(x): abs(x) +
                       width, :] = image[y:height, 0:width, :]
            elif x < 0 and x + patch_size > width:  # DONE
                canvas[:, abs(x):abs(x) + width,
                       :] = image[y:y + patch_size, 0:width, :]
            elif y < 0 and y + patch_size > height and x + patch_size > width:  # DONE
                canvas[abs(y):abs(y) + height, 0:width - x,
                       :] = image[0:height, x:width, :]
            elif y < 0 and x + patch_size > width:  # DONE
                canvas[abs(y):patch_size, 0:width - x,
                       :] = image[0:y + patch_size, x:width, :]
            elif y < 0 and y + patch_size > height:  # DONE
                canvas[abs(y):abs(y) + height, :,
                       :] = image[0:height, x:x + patch_size, :]
            elif y < 0:
                canvas[abs(y):patch_size, :, :] = image[0:y +
                                                        patch_size, x:x + patch_size, :]
            elif x + patch_size > width and y + patch_size > height:
                canvas[0:height - y, 0:width - x,
                       :] = image[y:height, x:width, :]
            elif x < 0 and y + patch_size > height:
                canvas[0:height - y, abs(x):patch_size,
                       :] = image[y:height, 0:x + patch_size, :]
            elif y + patch_size > height:
                canvas[0: height - y, :, :] = image[y:height, x:x + patch_size, :]
            elif x + patch_size > width:
                canvas[:, 0: width - x, :] = image[y:y + patch_size, x:width, :]
            elif x < 0:
                canvas[:, abs(x):patch_size, :] = image[y:y +
                                                        patch_size, 0:x + patch_size, :]
            else:
                canvas[:, :, :] = image[y:y + patch_size, x:x + patch_size, :]
        except Exception as e:
            print(e)
            print(
                f"X: {x}, Y: {y}, patch_size: {patch_size}, width: {width}, height: {height}")
            raise Exception(e)

        return canvas

    def read_image_patch(self, image, x, y, scale=1, patch_size=512, annotation_classes=None, use_cells=False):
        name = image['name']

        if self.inference:
            mask_patch = np.zeros(
                (patch_size, patch_size, len(annotation_classes)))
        else:
            mask_patch = self.mask_creator.create_mask_from_DB(
                image=image, patch=(x, y, patch_size),
                annotation_classes=annotation_classes
            )

        if self.directory is None:
            image_path = os.path.join('data', f"{name}_{y}_{x}_{scale}.tif")
            other_path = os.path.join(
                'data', f"{name}_{y}_{x}_{scale}_others.tif")
            immune_cell_path = os.path.join(
                'data', f"{name}_{y}_{x}_{scale}_immune_cells.tif")
            muscle_path = os.path.join(
                'data', f"{name}_{y}_{x}_{scale}_muscle.tif")
        else:
            image_path = os.path.join(
                self.directory, f"{name}_{y}_{x}_{scale}.tif")
            other_path = os.path.join(
                self.directory, f"{name}_{y}_{x}_{scale}_others.tif")
            immune_cell_path = os.path.join(
                self.directory, f"{name}_{y}_{x}_{scale}_immune_cells.tif")
            muscle_path = os.path.join(
                self.directory, f"{name}_{y}_{x}_{scale}_muscle.tif")

        img = load_img(image_path, color_mode='rgb')
        img = np.array(img) / 255

        if use_cells:
            other_img = load_img(other_path, color_mode='grayscale')
            immune_cells_img = load_img(
                immune_cell_path, color_mode='grayscale')
            muscle_img = load_img(muscle_path, color_mode='grayscale')

            immune_cells_img = np.expand_dims(immune_cells_img, 2)
            muscle_img = np.expand_dims(muscle_img, 2)
            other_img = np.expand_dims(other_img, 2)

            img = np.concatenate([
                img,
                immune_cells_img,
                muscle_img,
                other_img
            ], 2)

        return img, mask_patch

    def read_patch(self, image, fragment, x, y, patch_size=512, annotation_classes=None):
        name = image['name']

        fragment_x = fragment['x']
        fragment_y = fragment['y']
        fragment_width = fragment['width']
        fragment_height = fragment['height']

        if self.inference:
            mask_patch = np.zeros(
                (patch_size, patch_size, len(annotation_classes)))
        else:
            mask_patch = self.mask_creator.create_mask_from_DB(
                image=image,
                patch=(x, y, patch_size),
                fragment=(fragment_x, fragment_y,
                          fragment_width, fragment_height),
                annotation_classes=annotation_classes
            )

        x0_scale = x - fragment_x
        y0_scale = y - fragment_y

        if self.directory is None:
            img_scale0 = os.path.join(
                'data', f"{name}_{fragment['idx']}_0.npy")
        else:
            img_scale0 = os.path.join(
                self.directory, f"{name}_{fragment['idx']}_0.npy")

        img0_npy = np.load(img_scale0, mmap_mode='r')
        img_scale0 = self._get_fragment(
            img0_npy,
            x0_scale,
            y0_scale,
            patch_size
        )

        return img_scale0 / 255, mask_patch

    def read_patch_srel(self, image, fragment, x, y, patch_size=512, annotation_classes=None, lab=False):
        name = image['name']

        fragment_x = fragment['x']
        fragment_y = fragment['y']
        fragment_width = fragment['width']
        fragment_height = fragment['height']

        mask_patch = self.mask_creator.create_mask_from_DB(
            image=image,
            patch=(x - patch_size // 2, y - patch_size // 2, patch_size * 2),
            fragment=(fragment_x, fragment_y, fragment_width, fragment_height),
            annotation_classes=annotation_classes,
            factor=0.5
        )

        x0_scale = x // 2 - fragment_x
        y0_scale = y // 2 - fragment_y
        x1_scale = x0_scale - patch_size // 4
        y1_scale = y0_scale - patch_size // 4

        if self.directory is None:
            img_scale0 = os.path.join(
                'data', f"{name}_{fragment['idx']}_1.npy")
        else:
            img_scale0 = os.path.join(
                self.directory, f"{name}_{fragment['idx']}_1.npy")

        img0_npy = np.load(img_scale0, mmap_mode='r')
        img_scale0 = self._get_fragment(
            img0_npy,
            x1_scale,
            y1_scale,
            patch_size
        )
        del img0_npy
        img_scale0 /= 255

        if lab:
            img_scale0 = cv.cvtColor(
                img_scale0.astype('float32'), cv.COLOR_RGB2Lab)
            img_scale0 = normalize_lab(img_scale0)

        return img_scale0, mask_patch

    def read_in3_out1(self, image, fragment, x, y, patch_size=512, annotation_classes=None, use_cells=False):
        name = image['name']

        fragment_x = fragment['x']
        fragment_y = fragment['y']
        fragment_width = fragment['width']
        fragment_height = fragment['height']

        if self.inference:
            mask_patch = np.zeros(
                (patch_size, patch_size, len(annotation_classes)))
        else:
            mask_patch = self.mask_creator.create_mask_from_DB(
                image=image,
                patch=(x, y, patch_size),
                fragment=(fragment_x, fragment_y,
                          fragment_width, fragment_height),
                annotation_classes=annotation_classes
            )

        x0_scale = x - fragment_x
        x1_scale = int((x0_scale / 2) -
                       (patch_size / 2 - patch_size / 4))  # 128
        x2_scale = int((x0_scale / 4) -
                       (patch_size / 4 + patch_size / 8))  # 192

        y0_scale = y - fragment_y
        y1_scale = int((y0_scale / 2) - (patch_size / 2 - patch_size / 4))
        y2_scale = int((y0_scale / 4) - (patch_size / 4 + patch_size / 8))

        if self.directory is None:
            img_scale0 = os.path.join(
                'data/images', f"{name}_{fragment['idx']}_0.npy")
            img_scale1 = os.path.join(
                'data/images', f"{name}_{fragment['idx']}_1.npy")
            img_scale2 = os.path.join(
                'data/images', f"{name}_{fragment['idx']}_2.npy")
        else:
            img_scale0 = os.path.join(
                f'{self.directory}/images', f"{name}_{fragment['idx']}_0.npy")
            img_scale1 = os.path.join(
                f'{self.directory}/images', f"{name}_{fragment['idx']}_1.npy")
            img_scale2 = os.path.join(
                f'{self.directory}/images', f"{name}_{fragment['idx']}_2.npy")
            cell_path = os.path.join(
                f'{self.directory}/cells', f"{name}_{fragment['idx']}_0.npy")

        img0_npy = np.load(img_scale0, mmap_mode='r')
        img_scale0 = self._get_fragment(
            img0_npy,
            x0_scale,
            y0_scale,
            patch_size
        )
        del img0_npy

        img1_npy = np.load(img_scale1, mmap_mode='r')
        img_scale1 = self._get_fragment(
            img1_npy,
            x1_scale,
            y1_scale,
            patch_size
        )
        del img1_npy

        img2_npy = np.load(img_scale2, mmap_mode='r')
        img_scale2 = self._get_fragment(
            img2_npy,
            x2_scale,
            y2_scale,
            patch_size
        )
        del img2_npy

        img_scale0 = img_scale0 / 255
        img_scale1 = img_scale1 / 255
        img_scale2 = img_scale2 / 255

        if use_cells:
            img_cell = np.load(
                cell_path,
                mmap_mode='r'
            )

            img_scale0 = np.concatenate([
                img_scale0,
                self._get_fragment(np.expand_dims(
                    img_cell, 2), x0_scale, y0_scale, patch_size, fill_value=0)
            ], axis=-1)

        return img_scale0, img_scale1, img_scale2, mask_patch

    def read_in2_out2(self, image, fragment, x, y, patch_size=512, annotation_classes=None, use_cells=False, lab=False):
        name = image['name']

        fragment_x = fragment['x']
        fragment_y = fragment['y']
        fragment_width = fragment['width']
        fragment_height = fragment['height']

        if self.inference:
            mask_patch3 = np.zeros(
                (patch_size * 4, patch_size * 4, len(annotation_classes)))
        else:
            mask_patch3 = self.mask_creator.create_mask_from_DB(
                image=image,
                patch=(int(x - patch_size // 2) - patch_size,
                       int(y - patch_size // 2) - patch_size, patch_size * 4),
                fragment=(fragment_x, fragment_y,
                          fragment_width, fragment_height),
                annotation_classes=annotation_classes
            )

        mask_patch2 = mask_patch3[patch_size:patch_size *
                                  3, patch_size:patch_size * 3]
        mask_patch1 = mask_patch3[
            patch_size + patch_size // 2: 2 * patch_size + patch_size // 2,
            patch_size + patch_size // 2: 2 * patch_size + patch_size // 2
        ]

        x0 = x - fragment_x
        y0 = y - fragment_y

        x1 = int(x0 - patch_size // 2)
        y1 = int(y0 - patch_size // 2)

        if self.directory is None:
            img_path = os.path.join(
                'data/images', f"{name}_{fragment['idx']}_0.npy")
            cell_path = os.path.join(
                'data/cells', f"{name}_{fragment['idx']}_0.npy")
        else:
            img_path = os.path.join(
                f'{self.directory}/images', f"{name}_{fragment['idx']}_0.npy")
            cell_path = os.path.join(
                f'{self.directory}/cells', f"{name}_{fragment['idx']}_0.npy")

        img = np.load(
            img_path,
            mmap_mode='r'
        )

        if lab:
            image_patch1 = self._get_fragment(img, x0, y0, patch_size)
            image_patch2 = self._get_fragment(img, x1, y1, patch_size * 2)
            image_patch1 = cv.cvtColor(
                image_patch1.astype(np.uint8), cv.COLOR_RGB2LAB)
            image_patch2 = cv.cvtColor(
                image_patch2.astype(np.uint8), cv.COLOR_RGB2LAB)
        else:
            image_patch1 = self._get_fragment(img, x0, y0, patch_size) / 255
            image_patch2 = self._get_fragment(
                img, x1, y1, patch_size * 2) / 255

        if use_cells:
            img_cell = np.load(
                cell_path,
                mmap_mode='r'
            )
            image_patch1 = np.concatenate([
                image_patch1,
                self._get_fragment(np.expand_dims(
                    img_cell, 2), x0, y0, patch_size, fill_value=0)
            ], axis=-1)
            image_patch2 = np.concatenate([
                image_patch2,
                self._get_fragment(np.expand_dims(
                    img_cell, 2), x1, y1, patch_size * 2, fill_value=0)
            ], axis=-1)

        return image_patch1, image_patch2, mask_patch1, mask_patch2

    def read_in3_out3(self, image, fragment, x, y, patch_size=512, annotation_classes=None, use_cells=False):
        name = image['name']

        fragment_x = fragment['x']
        fragment_y = fragment['y']
        fragment_width = fragment['width']
        fragment_height = fragment['height']

        if self.inference:
            mask_patch3 = np.zeros(
                (patch_size * 4, patch_size * 4, len(annotation_classes)))
        else:
            mask_patch3 = self.mask_creator.create_mask_from_DB(
                image=image,
                patch=(int(x - patch_size // 2) - patch_size,
                       int(y - patch_size // 2) - patch_size, patch_size * 4),
                fragment=(fragment_x, fragment_y,
                          fragment_width, fragment_height),
                annotation_classes=annotation_classes
            )

        mask_patch2 = mask_patch3[patch_size:patch_size *
                                  3, patch_size:patch_size * 3]
        mask_patch1 = mask_patch3[
            patch_size + patch_size // 2: 2 * patch_size + patch_size // 2,
            patch_size + patch_size // 2: 2 * patch_size + patch_size // 2
        ]

        if self.directory is None:
            img_path = os.path.join(
                'data/images', f"{name}_{fragment['idx']}_0.npy")
            cell_path = os.path.join(
                'data/cells', f"{name}_{fragment['idx']}_0.npy")
        else:
            img_path = os.path.join(
                f'{self.directory}/images', f"{name}_{fragment['idx']}_0.npy")
            cell_path = os.path.join(
                f'{self.directory}/cells', f"{name}_{fragment['idx']}_0.npy")

        x0 = x - fragment_x
        y0 = y - fragment_y

        x1 = int(x0 - patch_size // 2)
        y1 = int(y0 - patch_size // 2)

        x2 = x1 - patch_size
        y2 = y1 - patch_size

        img = np.load(
            img_path,
            mmap_mode='r'
        )
        image_patch1 = self._get_fragment(img, x0, y0, patch_size)
        image_patch2 = self._get_fragment(img, x1, y1, patch_size * 2)
        image_patch3 = self._get_fragment(img, x2, y2, patch_size * 4)

        if use_cells:
            img_cell = np.load(
                cell_path,
                mmap_mode='r'
            )
            image_patch1 = np.concatenate([
                image_patch1,
                self._get_fragment(np.expand_dims(
                    img_cell, 2), x0, y0, patch_size, fill_value=0)
            ], axis=-1)
            image_patch2 = np.concatenate([
                image_patch2,
                self._get_fragment(np.expand_dims(
                    img_cell, 2), x1, y1, patch_size * 2, fill_value=0)
            ], axis=-1)
            image_patch3 = np.concatenate([
                image_patch3,
                self._get_fragment(np.expand_dims(
                    img_cell, 2), x2, y2, patch_size * 4, fill_value=0)
            ], axis=-1)

        return image_patch1, image_patch2, image_patch3, mask_patch1, mask_patch2, mask_patch3

    def read_in2_out1(self, image, fragment, x, y, patch_size=512, annotation_classes=None, use_cells=False):
        name = image['name']

        fragment_x = fragment['x']
        fragment_y = fragment['y']
        fragment_width = fragment['width']
        fragment_height = fragment['height']

        if self.inference:
            mask_patch = np.zeros(
                (patch_size, patch_size, len(annotation_classes)))
        else:
            mask_patch = self.mask_creator.create_mask_from_DB(
                image=image, patch=(x, y, patch_size),
                fragment=(fragment_x, fragment_y,
                          fragment_width, fragment_height),
                annotation_classes=annotation_classes
            )

        x0_scale = x - fragment_x
        x1_scale = int(x0_scale // 2)

        y0_scale = y - fragment_y
        y1_scale = int(y0_scale // 2)

        if self.directory is None:
            img_scale0 = os.path.join(
                'data', f"{name}_{fragment['idx']}_0.npy")
            img_scale1 = os.path.join(
                'data', f"{name}_{fragment['idx']}_1.npy")
            cell_path = os.path.join(
                'data/cells', f"{name}_{fragment['idx']}_0.npy")
        else:
            img_scale0 = os.path.join(
                f'{self.directory}/images', f"{name}_{fragment['idx']}_0.npy")
            img_scale1 = os.path.join(
                f'{self.directory}/images', f"{name}_{fragment['idx']}_1.npy")
            cell_path = os.path.join(
                f'{self.directory}/cells', f"{name}_{fragment['idx']}_0.npy")

        img0_npy = np.load(img_scale0, mmap_mode='r')
        img_scale0 = self._get_fragment(
            img0_npy,
            x0_scale,
            y0_scale,
            patch_size
        )
        del img0_npy

        img1_npy = np.load(img_scale1, mmap_mode='r')
        img_scale1 = self._get_fragment(
            img1_npy,
            x1_scale,
            y1_scale,
            patch_size // 2
        )
        del img1_npy

        img_scale0 = img_scale0 / 255

        if use_cells:
            img_cell = np.load(
                cell_path,
                mmap_mode='r'
            )

            img_scale0 = np.concatenate([
                img_scale0,
                self._get_fragment(np.expand_dims(
                    img_cell, 2), x0_scale, y0_scale, patch_size, fill_value=0)
            ], axis=-1)

        return img_scale0, img_scale1 / 255, mask_patch

    def read_in1_out3(self, image, fragment, x, y, patch_size=512, annotation_classes=None, use_cells=False, lab=False):
        name = image['name']

        fragment_x = fragment['x']
        fragment_y = fragment['y']
        fragment_width = fragment['width']
        fragment_height = fragment['height']

        if self.inference:
            mask_patch = np.zeros(
                (patch_size, patch_size, len(annotation_classes)))
        else:
            mask_patch = self.mask_creator.create_mask_from_DB(
                image=image,
                patch=(x, y, patch_size),
                fragment=(fragment_x, fragment_y,
                          fragment_width, fragment_height),
                annotation_classes=annotation_classes
            )

        x0_scale = x - fragment_x
        y0_scale = y - fragment_y

        if self.directory is None:
            img_scale0 = os.path.join(
                'data', f"{name}_{fragment['idx']}_0.npy")
            cell_path = os.path.join(
                'data/cells', f"{name}_{fragment['idx']}_0.npy")
        else:
            img_scale0 = os.path.join(
                self.directory, f"{name}_{fragment['idx']}_0.npy")
            cell_path = os.path.join(
                f'{self.directory}/cells', f"{name}_{fragment['idx']}_0.npy")

        img0_npy = np.load(img_scale0, mmap_mode='r')
        img_scale0 = self._get_fragment(
            img0_npy,
            x0_scale,
            y0_scale,
            patch_size
        )

        img_scale0 = img_scale0 / 255

        if use_cells:
            img_cell = np.load(
                cell_path,
                mmap_mode='r'
            )

            img_scale0 = np.concatenate([
                img_scale0,
                self._get_fragment(np.expand_dims(
                    img_cell, 2), x0_scale, y0_scale, patch_size, fill_value=0)
            ], axis=-1)

        return img_scale0, np.expand_dims(mask_patch[:, :, 0], -1), np.expand_dims(mask_patch[:, :, 1], -1), \
            np.expand_dims(mask_patch[:, :, 2], -1)

    def read_multiscale_patch1_file(self, image, fragment, x, y, patch_size=512, annotation_classes=None,
                                    use_cells=False):
        name = image['name']

        fragment_x = fragment['x']
        fragment_y = fragment['y']
        fragment_width = fragment['width']
        fragment_height = fragment['height']
        fragment_idx = fragment['idx']

        mask_patch3 = self.mask_creator.create_mask_from_file(
            image=image,
            fragment=(
                fragment_x,
                fragment_y,
                fragment_width,
                fragment_height,
                fragment_idx
            ),
            patch=(int(x - patch_size // 2) - patch_size,
                   int(y - patch_size // 2) - patch_size, patch_size * 4),
            annotation_classes=annotation_classes
        )

        mask_patch2 = mask_patch3[patch_size:patch_size *
                                  3, patch_size:patch_size * 3]
        mask_patch1 = mask_patch3[
            patch_size + patch_size // 2: 2 * patch_size + patch_size // 2,
            patch_size + patch_size // 2: 2 * patch_size + patch_size // 2
        ]

        x0 = x - fragment_x
        y0 = y - fragment_y

        x1 = int(x0 - patch_size // 2)
        y1 = int(y0 - patch_size // 2)

        x2 = x1 - patch_size
        y2 = y1 - patch_size

        if self.directory is None:
            img_path = os.path.join(
                'data/images', f"{name}_{fragment['idx']}_0.npy")
            cell_path = os.path.join(
                'data/cells', f"{name}_{fragment['idx']}_0.npy")
        else:
            img_path = os.path.join(
                f'{self.directory}/images', f"{name}_{fragment['idx']}_0.npy")
            cell_path = os.path.join(
                f'{self.directory}/cells', f"{name}_{fragment['idx']}_0.npy")

        img = np.load(
            img_path,
            mmap_mode='r'
        )

        image_patch1 = self._get_fragment(img, x0, y0, patch_size) / 255
        image_patch2 = self._get_fragment(img, x1, y1, patch_size * 2) / 255
        image_patch3 = self._get_fragment(img, x2, y2, patch_size * 4) / 255

        if use_cells:
            img_cell = np.load(
                cell_path,
                mmap_mode='r'
            )
            image_patch1 = np.concatenate([
                image_patch1,
                self._get_fragment(np.expand_dims(
                    img_cell, 2), x0, y0, patch_size)
            ], axis=-1)
            image_patch2 = np.concatenate([
                image_patch2,
                self._get_fragment(np.expand_dims(
                    img_cell, 2), x1, y1, patch_size * 2)
            ], axis=-1)
            image_patch3 = np.concatenate([
                image_patch3,
                self._get_fragment(np.expand_dims(
                    img_cell, 2), x2, y2, patch_size * 4)
            ], axis=-1)

        return image_patch1, image_patch2, image_patch3, mask_patch1, mask_patch2, mask_patch3

    def read_multiscale_patch3_file(self, image, fragment, x, y, patch_size=512, annotation_classes=None):
        name = image['name']

        fragment_x = fragment['x']
        fragment_y = fragment['y']
        fragment_width = fragment['width']
        fragment_height = fragment['height']
        fragment_idx = fragment['idx']

        mask_patch = self.mask_creator.create_mask_from_file(
            image=image,
            patch=(x, y, patch_size),
            fragment=(fragment_x, fragment_y, fragment_width,
                      fragment_height, fragment_idx),
            annotation_classes=annotation_classes
        )

        x0_scale = x - fragment_x
        x1_scale = int((x0_scale / 2) -
                       (patch_size / 2 - patch_size / 4))  # 128
        x2_scale = int((x0_scale / 4) -
                       (patch_size / 4 + patch_size / 8))  # 192

        y0_scale = y - fragment_y
        y1_scale = int((y0_scale / 2) - (patch_size / 2 - patch_size / 4))
        y2_scale = int((y0_scale / 4) - (patch_size / 4 + patch_size / 8))

        if self.directory is None:
            img_scale0 = os.path.join(
                'data', f"{name}_{fragment['idx']}_0.npy")
            img_scale1 = os.path.join(
                'data', f"{name}_{fragment['idx']}_1.npy")
            img_scale2 = os.path.join(
                'data', f"{name}_{fragment['idx']}_2.npy")
        else:
            img_scale0 = os.path.join(
                self.directory, f"{name}_{fragment['idx']}_0.npy")
            img_scale1 = os.path.join(
                self.directory, f"{name}_{fragment['idx']}_1.npy")
            img_scale2 = os.path.join(
                self.directory, f"{name}_{fragment['idx']}_2.npy")

        img0_npy = np.load(img_scale0, mmap_mode='r')
        img_scale0 = self._get_fragment(
            img0_npy,
            x0_scale,
            y0_scale,
            patch_size
        )
        del img0_npy

        img1_npy = np.load(img_scale1, mmap_mode='r')
        img_scale1 = self._get_fragment(
            img1_npy,
            x1_scale,
            y1_scale,
            patch_size
        )
        del img1_npy

        img2_npy = np.load(img_scale2, mmap_mode='r')
        img_scale2 = self._get_fragment(
            img2_npy,
            x2_scale,
            y2_scale,
            patch_size
        )
        del img2_npy

        return img_scale0 / 255, img_scale1 / 255, img_scale2 / 255, mask_patch
