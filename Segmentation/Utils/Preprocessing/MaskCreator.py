import os.path

import numpy as np
import cv2 as cv
import json

from .HasuraClient import HasuraClient
from .Database import Database
from .Geopandas import Geopandas


class MaskCreator:
    def __init__(self, database_type, url=None, headers=None, out_dir=None):
        self.out_dir = out_dir
        if database_type == 'hasura':
            if url is None:
                self.client = None
            else:
                self.client = HasuraClient(url, headers)
        elif database_type == 'postgres':
            self.client = Database()
        elif database_type == 'files':
            self.client = Geopandas()

    def _get_fragment(self, image, x, y, patch_size, fill_value=0, is_one_channel=True):
        if is_one_channel:
            height, width = image.shape
            canvas = np.full((patch_size, patch_size), fill_value=fill_value)
        else:
            height, width, classes = image.shape
            canvas = np.full((patch_size, patch_size, classes),
                             fill_value=fill_value)

        if x < 0 and y < 0 and x + patch_size > width and y + patch_size > height:  # DONE
            canvas[abs(y):abs(y) + height, abs(x): abs(x) +
                   width] = image[0:height, 0:width]
        elif x < 0 and y < 0 and x + patch_size > width:  # DONE
            canvas[abs(y):patch_size, abs(x):abs(x) +
                   width] = image[0:y + patch_size, 0:width]
        elif x < 0 and y < 0 and y + patch_size > height:  # DONE
            canvas[abs(y):abs(y) + height, abs(x)
                       :patch_size] = image[0:height, 0:x + patch_size]
        elif x < 0 and y < 0:  # DONE
            canvas[abs(y):patch_size, abs(
                x):patch_size] = image[0: y + patch_size, 0: x + patch_size]
        elif x < 0 and x + patch_size > width and y + patch_size > height:  # DONE
            canvas[0:height - y, abs(x): abs(x) +
                   width] = image[y:height, 0:width]
        elif x < 0 and x + patch_size > width:  # DONE
            canvas[:, abs(x):abs(x) + width] = image[y:y + patch_size, 0:width]
        elif y < 0 and y + patch_size > height and x + patch_size > width:  # DONE
            canvas[abs(y):abs(y) + height, 0:width -
                   x] = image[0:height, x:width]
        elif y < 0 and x + patch_size > width:  # DONE
            canvas[abs(y):patch_size, 0:width -
                   x] = image[0:y + patch_size, x:width]
        elif y < 0 and y + patch_size > height:  # DONE
            canvas[abs(y):abs(y) + height,
                   :] = image[0:height, x:x + patch_size]
        elif y < 0:
            canvas[abs(y):patch_size, :] = image[0:y +
                                                 patch_size, x:x + patch_size]
        elif x + patch_size > width and y + patch_size > height:
            canvas[0:height - y, 0:width - x] = image[y:height, x:width]
        elif x < 0 and y + patch_size > height:
            canvas[0:height - y,
                   abs(x):patch_size] = image[y:height, 0:x + patch_size]
        elif y + patch_size > height:
            canvas[0: height - y, :] = image[y:height, x:x + patch_size]
        elif x + patch_size > width:
            canvas[:, 0: width - x] = image[y:y + patch_size, x:width]
        elif x < 0:
            canvas[:, abs(x):patch_size] = image[y:y +
                                                 patch_size, 0:x + patch_size]
        else:
            canvas = image[y:y + patch_size, x:x + patch_size]

        return canvas

    def create_mask_from_DB(self, image, directory=None, npy_name=None, patch=None, annotation_classes=None, fragment=None, factor=1):
        if annotation_classes is None:
            annotation_classes = [
                'blood_vessels',
                'endocariums',
                'fatty_tissues',
                'fibrotic_tissues',
                'immune_cells',
                'inflammations',
                'quilties'
            ]

        if self.client is None:
            print('Missing DB information!')
            return

        name = image['name']
        height = int(image['height'] * factor)
        width = int(image['width'] * factor)

        if patch is None:
            annotations = self.client.get_annotations_by_name(
                file_name=name,
                annotation_classes=annotation_classes
            )
        else:
            x, y, size = patch

            annotations = self.client.get_patch_annotations_by_name(
                file_name=name,
                annotation_classes=annotation_classes,
                patch=patch
            )

        if annotations == 0:
            if patch is None:
                return np.zeros((height, width, len(annotation_classes)), dtype=np.int8)
            else:
                size = int(size * factor)
                return np.zeros((size, size, len(annotation_classes)), dtype=np.int8)

        if annotations is None:
            print("Annotations do not exist!")
            return

        if patch is None:
            final_mask = np.zeros(
                (height, width, len(annotation_classes)), dtype=np.int8)
        else:
            size = int(size * factor)
            final_mask = np.zeros(
                (size, size, len(annotation_classes)), dtype=np.int8)

        x, y = int(x * factor), int(y * factor)
        for idx, annotation_class in enumerate(annotation_classes):
            if patch is not None:
                annotation_class += '_patch'

            canvas = np.zeros((width, height), dtype=np.int8)
            for annotation in annotations[annotation_class]:
                for coors in annotation['scaled_annotation']['coordinates']:
                    if annotation['scaled_annotation']['type'] == 'MultiPolygon':
                        coors = coors[0]

                    pts = [[round(c[1] * factor), round(c[0] * factor)]
                           for c in coors]
                    cv.fillPoly(canvas, [np.array(pts)], 1)

            canvas = canvas.transpose()
            if patch is None:
                final_mask[:, :, idx] = canvas
            else:
                if fragment is not None:
                    frag_x, frag_y, frag_width, frag_height = fragment
                    fragment_patch = canvas[frag_y:frag_y +
                                            frag_height, frag_x:frag_x + frag_width]
                    if x - frag_x < 0 or y - frag_y < 0 or x - frag_x + size > frag_width or y - frag_y + size > frag_height:
                        final_mask[:, :, idx] = self._get_fragment(
                            fragment_patch, x - frag_x, y - frag_y, size)
                    else:
                        final_mask[:, :, idx] = fragment_patch[y - frag_y:y -
                                                               frag_y + size, x - frag_x:x - frag_x + size]

                else:
                    if x < 0 or y < 0 or x + size > width or y + size > height:
                        final_mask[:, :, idx] = self._get_fragment(
                            canvas, x, y, size)
                    else:
                        final_mask[:, :, idx] = canvas[y:y + size, x:x + size]

        if directory is not None and npy_name is not None:
            np.save(directory + '/' + npy_name, final_mask)
        else:
            return final_mask
        return final_mask

    def create_masks_from_DB(self, json_file, data_type):
        if self.client is None:
            print('Missing DB information!')
            return

        for image in json_file['images']:
            name = image['name']
            height = image['height']
            width = image['width']

            canvas = np.zeros((width, height), dtype=np.int8)
            if data_type == 'endocariums':
                polygons = self.client.get_endocariums_by_name(name)
            elif data_type == 'fatty_tissues':
                polygons = self.client.get_fatty_tissues_by_name(name)
            elif data_type == 'blood_vessels':
                polygons = self.client.get_blood_vessels_by_name(name)
            elif data_type == 'inflammations':
                polygons = self.client.get_inflammations_by_name(name)
            else:
                polygons = []

            for polygon in polygons:
                for coors in polygon['annotation']['coordinates']:
                    if polygon['annotation']['type'] == 'MultiPolygon':
                        coors = coors[0]
                    pts = [[round(c[1]), round(c[0])] for c in coors]
                    cv.fillPoly(canvas, [np.array(pts)], 1)

            canvas = canvas.transpose()
            output_name = name + '-' + data_type + '.png'
            cv.imwrite(output_name, canvas * 255)

        print("EXPORT IS DONE!")

    def create_mask_from_GEOJSON(self, json_file, data_type, geojson_name, data_dir):
        name = geojson_name.replace(data_dir, '').replace(
            '/', '').replace('\\', '')
        index = name.index('.')
        name = name[:index]
        print(name)

        with open(geojson_name) as f:
            json_data = json.load(f)

        for image in json_file['images']:
            if image['name'] != name:
                continue

            height = image['height']
            width = image['width']

            canvas = np.zeros((width, height), dtype=np.int8)
            for feature in json_data['features']:
                if feature['geometry']['type'] == 'MultiPolygon':
                    continue
                if data_type is not None and feature['properties']['classification']['name'] != data_type:
                    continue
                polygon = feature['nucleusGeometry']
                for coors in polygon['coordinates']:
                    if polygon['type'] == 'MultiPolygon':
                        coors = coors[0]
                    pts = [[round(c[1]), round(c[0])] for c in coors]
                    cv.fillPoly(canvas, [np.array(pts)], 1)

            canvas = canvas.transpose()
            return canvas

    def create_mask_from_data(self, data, x_size, y_size):
        height = x_size
        width = y_size

        canvas = np.zeros((width, height), dtype=np.int8)

        for polygon in data:
            for coors in polygon['annotation']['coordinates']:
                if polygon['annotation']['type'] == 'MultiPolygon':
                    coors = coors[0]
                pts = [[round(c[1]), round(c[0])] for c in coors]
                cv.fillPoly(canvas, [np.array(pts)], 1)

        canvas = canvas.transpose()

        return canvas

    def create_mask_from_file(self, image, fragment, patch=None, annotation_classes=None):
        name = image['name']
        height = image['height']
        width = image['width']
        frag_x, frag_y, frag_width, frag_height, fragment_idx = fragment

        fragment_patch = np.load(
            f"{self.out_dir}/{name}_{fragment_idx}_0.npy",
            mmap_mode='r'
        )

        if patch is None:
            return fragment_patch
        else:
            x, y, size = patch

            if x - frag_x < 0 or y - frag_y < 0 or x - frag_x + size > frag_width or y - frag_y + size > frag_height:
                final_mask = self._get_fragment(
                    fragment_patch, x - frag_x, y - frag_y, size, is_one_channel=False)
            else:
                final_mask = fragment_patch[y - frag_y:y -
                                            frag_y + size, x - frag_x:x - frag_x + size]

        return final_mask
