import random
from PIL import Image
import os
import javabridge
import bioformats
import glob
import math
import json
import numpy as np
from tqdm import tqdm
import cv2 as cv
import geojson
from shapely.geometry import shape

from MaskCreator import MaskCreator


def stop_logging():
    logger_name = javabridge.get_static_field("org/slf4j/Logger",
                                              "ROOT_LOGGER_NAME",
                                              "Ljava/lang/String;")

    root_logger = javabridge.static_call("org/slf4j/LoggerFactory",
                                         "getLogger",
                                         "(Ljava/lang/String;)Lorg/slf4j/Logger;",
                                         logger_name)

    log_level = javabridge.get_static_field("ch/qos/logback/classic/Level",
                                            "WARN",
                                            "Lch/qos/logback/classic/Level;")

    javabridge.call(root_logger,
                    "setLevel",
                    "(Lch/qos/logback/classic/Level;)V",
                    log_level)


def save_image(image, name):
    img_pil = Image.fromarray(image, "RGB")
    img_pil.save(name)


class Patcher:
    def __init__(self, directory, annotation_directory, output_directory):
        super(Patcher, self).__init__()
        self.directory = directory
        self.annotation_directory = annotation_directory
        self.output_directory = output_directory
        self.mask_creator = MaskCreator(
            'files', None, None)

    def check_existing_annotation(self, file_name):
        file_name = file_name.replace(
            '.vsi', '').replace(f'{self.directory}\\', '')
        annotations = glob.glob(f'{self.annotation_directory}\\*.geojson')
        for annotation in annotations:
            if file_name in annotation:
                return True
        return False

    def create_patches(self, patch_size=512, step=512):
        javabridge.start_vm(class_path=bioformats.JARS)
        # stop_logging()

        paths = glob.glob(f'{self.directory}\\*.vsi')
        final_json = {
            "images": [],
            "patch_size": patch_size
        }

        for index, file in tqdm(enumerate(paths), total=len(paths)):
            # check if file contains HE
            if 'HE' in file and self.check_existing_annotation(file):
                # create patches of image
                img_json = self.create_file_patches(
                    file, patch_size=final_json['patch_size'], step=step)
                final_json['images'].append(img_json)

        json_string = json.dumps(final_json)
        with open('images.json', 'w') as outfile:
            outfile.write(json_string)

        javabridge.kill_vm()

    def create_file_patches(self, file_name, patch_size=512, scale=1, threshold=0.05, step=512):
        with bioformats.ImageReader(file_name) as reader:
            name = file_name.replace('.vsi', '').replace(
                f'{self.directory}\\', '')

            # get image width and height
            size_x = reader.rdr.getSizeX()
            size_y = reader.rdr.getSizeY()

            json_file = {
                "name": name,
                "height": size_y,
                "width": size_x,
                "channels": 3,
                "patches": []
            }

            # count number of patches on every axis
            x_patches = int(math.ceil(size_x / patch_size))
            y_patches = int(math.ceil(size_y / patch_size))
            x_patches += (x_patches - 1) * (patch_size / step - 1)
            y_patches += (y_patches - 1) * (patch_size / step - 1)

            # iterate over patches
            for y_patch in range(int(y_patches)):
                for x_patch in range(int(x_patches)):
                    # get x and y coors of the start of the patch
                    x = x_patch * step
                    y = y_patch * step

                    # check patch width and height
                    width = size_x - \
                        x if (x + patch_size > size_x) else patch_size
                    height = size_y - y if (
                        y + patch_size > size_y) else patch_size

                    # read patch from an image
                    patch = reader.read(
                        0, rescale=False, XYWH=(x, y, width, height))

                    gray_patch = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
                    binary_mask = cv.adaptiveThreshold(
                        gray_patch, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 7)
                    binary_mask = cv.morphologyEx(
                        binary_mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (50, 50)))
                    binary_mask = cv.bitwise_not(binary_mask)

                    #unique, counts = np.unique(binary_mask, return_counts=True)
                    mask = self.mask_creator.create_mask_from_DB(
                        image={
                            "name": name,
                            "height": size_y,
                            "width": size_x,
                        },
                        patch=(x, y, patch_size),
                        annotation_classes=[
                            "blood_vessels",
                            "endocariums",
                            "fatty_tissues",
                            "fibrotic_tissues",
                            "inflammations",
                            "quilties"
                        ]
                    )
                    # if len(counts) > 1 and counts[1] >= pixel_count * threshold:
                    if mask.any():
                        unique, count = np.unique(mask, return_counts=True)
                        if count[1] > 0.1 * patch_size * patch_size:
                            print(unique, count)
                            json_file['patches'].append((y_patch, x_patch))
                            # save image patch, file name contain x and y value of patch
                            output_name = os.path.join(
                                self.output_directory, f"{name}_{y}_{x}_{scale}.tif")
                            #save_image(patch, output_name)

        return json_file

    def _create_patches(self, name, image, image_type, x, y, patch_size=512, scale=1):
        size_x = image.shape[1]
        size_y = image.shape[0]

        # check patch width and height
        width = size_x - x if (x + patch_size > size_x) else patch_size
        height = size_y - y if (y + patch_size > size_y) else patch_size

        patch = image[y:y + height, x:x + width]

        output_name = os.path.join(
            self.output_directory, f"{name}_{y}_{x}_{scale}_{image_type}.tif")
        cv.imwrite(output_name, patch)

    def create_annotation_patches(self, step=512):
        f = open(
            'C:\\Users\\ivanv\\Desktop\\Ivan\\FIIT\\02_ING\\Master thesis\\Code\\Segmentation\\images.json')
        json_images = json.load(f)
        f.close()

        for image in json_images['images']:
            name = image['name']
            height = image['height']
            width = image['width']

            other_canvas = np.zeros((width, height), dtype=np.int8)
            immune_cells_canvas = np.zeros((width, height), dtype=np.int8)
            muscle_canvas = np.zeros((width, height), dtype=np.int8)

            f = open(
                f'C:\\Users\\ivanv\\Desktop\\Ivan\\FIIT\\02_ING\\Master thesis\\Code\\Segmentation\\annotations\\{name}.vsi - 20x.geojson')
            json_data = json.load(f)
            f.close()

            other_polygons = []
            immune_cells_polygons = []
            muscle_polygons = []
            for feature in json_data['features']:
                class_type = feature['properties']['classification']['name']
                geometry = feature.get('nucleusGeometry')
                if geometry is None:
                    continue

                if class_type == 'Immune cells':
                    immune_cells_polygons.append(geometry)
                elif class_type == 'Muscle cells':
                    muscle_polygons.append(geometry)
                elif class_type == 'Other cells':
                    other_polygons.append(geometry)

            for polygon in other_polygons:
                for coors in polygon['coordinates']:
                    if polygon['type'] == 'MultiPolygon':
                        coors = coors[0]

                    pts = [[round(c[1]), round(c[0])] for c in coors]
                    cv.fillPoly(other_canvas, [np.array(pts)], 1)

            for polygon in immune_cells_polygons:
                for coors in polygon['coordinates']:
                    if polygon['type'] == 'MultiPolygon':
                        coors = coors[0]

                    pts = [[round(c[1]), round(c[0])] for c in coors]
                    cv.fillPoly(immune_cells_canvas, [np.array(pts)], 1)

            for polygon in muscle_polygons:
                for coors in polygon['coordinates']:
                    if polygon['type'] == 'MultiPolygon':
                        coors = coors[0]

                    pts = [[round(c[1]), round(c[0])] for c in coors]
                    cv.fillPoly(muscle_canvas, [np.array(pts)], 1)

            for patch in image['patches']:
                y_patch, x_patch = patch

                x = x_patch * step
                y = y_patch * step

                self._create_patches(name, other_canvas.transpose(
                ), 'others', x, y, json_images['patch_size'])
                self._create_patches(name, immune_cells_canvas.transpose(
                ), 'immune_cells', x, y, json_images['patch_size'])
                self._create_patches(name, muscle_canvas.transpose(
                ), 'muscle', x, y, json_images['patch_size'])

    def _find_tissues(self, gj):
        features = gj['features']

        for feature in features:
            class_type = feature['properties']['classification']['name']
            if class_type == 'Region*':
                return feature

    def _get_fragments(self, file_name, scales):
        gj = geojson.load(open(file_name))
        tissues = self._find_tissues(gj)['geometry']['coordinates']
        len_fragments = len(tissues)

        tissues_coord = {}
        for idx in range(len(scales)):
            tissues_coord[idx] = []

        for scale_idx, scale in enumerate(scales):
            for idx in range(len_fragments):
                coors = tissues[idx]

                polygon = shape({'type': 'Polygon', 'coordinates': coors})

                bounds = polygon.bounds
                bounds = [int(x * scale) for x in bounds]
                tissues_coord[scale_idx].append(bounds)

        return tissues_coord

    def create_multi_scale_fragments(self, file_name, scales, patch_size=512):
        name = file_name.replace('.vsi', '').replace(f'{self.directory}\\', '')
        geojson_name = f"{self.annotation_directory}\\{name}.vsi - 20x.geojson"

        tissue_coors = self._get_fragments(geojson_name, scales)

        json_file = {
            "name": name,
            "channels": 3,
            "fragments": []
        }

        for scale_idx, scale in enumerate(scales):
            features = tissue_coors[scale_idx]

            image_reader = bioformats.formatreader.make_image_reader_class()()
            image_reader.allowOpenToCheckType(True)
            image_reader.setId(file_name)
            image_reader.setSeries(scale_idx)
            wrapper = bioformats.formatreader.ImageReader(
                path=file_name, perform_init=False)
            wrapper.rdr = image_reader

            if scale_idx == 0:
                size_x = wrapper.rdr.getSizeX()
                size_y = wrapper.rdr.getSizeY()
                json_file['height'] = size_y
                json_file['width'] = size_x

            for idx, (left, top, right, bottom) in tqdm(enumerate(features), total=len(features)):
                patch = wrapper.read(0, rescale=False, XYWH=(
                    left, top, right - left, bottom - top))
                img_PIL = Image.fromarray(patch, "RGB")
                # img_PIL.save(f"{self.output_directory}/{name}_{idx}_{scale_idx}.tiff")
                # np.save(
                #    f"{self.output_directory}/{name}_{idx}_{scale_idx}.npy", np.array(img_PIL))

                if scale_idx == 0:
                    x_patches = int(math.ceil((right - left) / patch_size))
                    y_patches = int(math.ceil((bottom - top) / patch_size))
                    x_patches += (x_patches - 1) * (patch_size / step - 1)
                    y_patches += (y_patches - 1) * (patch_size / step - 1)
                    fragment = {
                        'name': f"{name}_{idx}_{scale_idx}",
                        'x': left,
                        'y': top,
                        'width': right - left,
                        'height': bottom - top,
                        'patches': []
                    }

                    y = top
                    for y_patch in range(int(y_patches)):
                        x = left
                        for x_patch in range(int(x_patches)):
                            mask = self.mask_creator.create_mask_from_DB(
                                image={
                                    "name": name,
                                    "height": size_y,
                                    "width": size_x,
                                },
                                patch=(x, y, patch_size),
                                fragment=(left, top, right -
                                          left, bottom - top),
                                annotation_classes=[
                                    "blood_vessels",
                                    "endocariums",
                                    "fatty_tissues",
                                    "fibrotic_tissues",
                                    "inflammations",
                                    "quilties"
                                ]
                            )
                            # with probability 0.1 accept mask without any annotation
                            prob = 0.1
                            if mask.any() or random.random() < prob:
                                # unique, count = np.unique(
                                #    mask, return_counts=True)
                                # if count[1] > 0.10 * patch_size * patch_size:
                                fragment['patches'].append((y, x))

                            x += step

                        y += step

                    json_file['fragments'].append(fragment)

        return json_file


if __name__ == '__main__':
    javabridge.start_vm(class_path=bioformats.JARS)
    stop_logging()

    patch_size = 512
    step = 256
    directory = 'D:\\Master Thesis\\Data\\EMB-IKEM-2022-03-09'

    vsi_files = glob.glob(f'{directory}\\*.vsi')
    final_json = {
        "images": [],
        "patch_size": patch_size
    }

    patcher = Patcher(
        directory=directory,
        annotation_directory='D:\\Master Thesis\\Data\\EMB-IKEM-2022-03-09\\QuPath project EMB - anotations\\srel annotations',
        output_directory='D:\\Master Thesis\\Code\\Segmentation\\data5'
    )

    for index, file in enumerate(vsi_files):
        # check if file contains HE
        if 'HE' in file:
            print(f"Creating fragments for {file}")
            # create patches of image
            img_json = patcher.create_multi_scale_fragments(
                file_name=file,
                patch_size=patch_size,
                scales=[1, 0.5, 0.25]
            )
            final_json['images'].append(img_json)

    json_string = json.dumps(final_json)
    with open(f"D:/Master Thesis/Code/Segmentation/data5/images_{patch_size}_{step}.json", 'w') as outfile:
        outfile.write(json_string)

    javabridge.kill_vm()
