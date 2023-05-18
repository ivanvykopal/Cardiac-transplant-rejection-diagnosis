import geopandas as gpd
import tensorflow as tf
import argparse
import numpy as np
import cv2 as cv
import javabridge
import bioformats as bf
import math
import albumentations as A
import logging
import json
from os.path import exists
import sys

from utils.utils import read_config, get_gauss, reorder_channels, applicate_augmentations, get_cells, get_fragment
from utils.create_geojson import create_geojson
from utils.post_process import postprocess_mask, postprocess_endocard, postprocess_inflammation, postprocess_vessels, merge_nearest_endocards
from models.utils import get_model
from utils.Dataset import Dataset

import warnings
warnings.simplefilter("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('final_cycle')


def set_logs():
    myloglevel = "ERROR"
    rootLoggerName = javabridge.get_static_field(
        "org/slf4j/Logger", "ROOT_LOGGER_NAME", "Ljava/lang/String;")
    rootLogger = javabridge.static_call("org/slf4j/LoggerFactory", "getLogger",
                                        "(Ljava/lang/String;)Lorg/slf4j/Logger;", rootLoggerName)
    logLevel = javabridge.get_static_field("ch/qos/logback/classic/Level", myloglevel,
                                           "Lch/qos/logback/classic/Level;")
    javabridge.call(rootLogger, "setLevel",
                    "(Lch/qos/logback/classic/Level;)V", logLevel)


def handle_data(reader, image1, config, coordinates, cell_mask, size):
    x, y, x_patch, y_patch = coordinates
    size_x, size_y = size
    patch_size = config['image_size']
    lab = config['lab']

    if config['output_masks'] == 1:
        if config['input_images'] == 1:
            if image1.shape[0] != patch_size or image1.shape[1] != patch_size:
                image1 = get_fragment(
                    image1, x, y, patch_size, fill_value=255.).astype('float')
            image1 /= 255
            return [image1]
        elif config['input_images'] == 2:
            reader.rdr.setSeries(1)
            if y + y_patch > size_y:
                y_patch = size_y - y
            if x + x_patch > size_x:
                x_patch = size_x - x
            image2 = reader.read(0, rescale=False, XYWH=(
                x // 2, y // 2, x_patch // 2, y_patch // 2)).astype('float')
            image1 /= 255
            image2 /= 255

            if cell_mask is not None:
                cell_mask_patch = cell_mask[y:y + y_patch, x:x + x_patch]

                if image1.shape[0] != patch_size or image1.shape[1] != patch_size:
                    image1 = get_fragment(
                        image1, x, y, patch_size, fill_value=1.).astype('float')
                    cell_mask_patch = get_fragment(
                        cell_mask_patch, x, y, patch_size, fill_value=0.)
                if image2.shape[0] != patch_size // 2 or image2.shape[1] != patch_size // 2:
                    image2 = get_fragment(
                        image2, x // 2, y // 2, patch_size // 2, fill_value=1.).astype('float')

                image1 = np.concatenate([
                    image1,
                    np.expand_dims(cell_mask_patch, -1)
                ], axis=-1)
            return [image1, image2]
        elif config['input_images'] == 3:
            x1_scale = int((x / 2) - (patch_size / 2 - patch_size / 4))  # 128
            x2_scale = int((x / 4) - (patch_size / 4 + patch_size / 8))  # 192

            y1_scale = int((y / 2) - (patch_size / 2 - patch_size / 4))
            y2_scale = int((y / 4) - (patch_size / 4 + patch_size / 8))

            x1_scale2 = x1_scale if x1_scale > 0 else 0
            x2_scale2 = x2_scale if x2_scale > 0 else 0

            y1_scale2 = y1_scale if y1_scale > 0 else 0
            y2_scale2 = y2_scale if y2_scale > 0 else 0

            if y1_scale + patch_size > size_y // 2:
                y_patch = size_y // 2 - y1_scale
            if x1_scale + patch_size > size_x // 2:
                x_patch = size_x // 2 - x1_scale

            reader.rdr.setSeries(1)
            image2 = reader.read(0, rescale=False, XYWH=(
                x1_scale2, y1_scale2, x_patch, y_patch)).astype('float')

            if y2_scale + patch_size > size_y // 4:
                y_patch = size_y // 4 - y2_scale
            if x2_scale + patch_size > size_x // 4:
                x_patch = size_x // 4 - x2_scale

            reader.rdr.setSeries(2)
            image3 = reader.read(0, rescale=False, XYWH=(
                x2_scale2, y2_scale2, x_patch, y_patch)).astype('float')

            if image1.shape[0] != patch_size or image1.shape[1] != patch_size:
                image1 = get_fragment(
                    image1, x, y, patch_size, fill_value=255.).astype('float')
            if image2.shape[0] != patch_size or image2.shape[1] != patch_size:
                image2 = get_fragment(
                    image2, x1_scale, y1_scale, patch_size, fill_value=255.).astype('float')
            if image3.shape[0] != patch_size or image3.shape[1] != patch_size:
                image3 = get_fragment(
                    image3, x2_scale, y2_scale, patch_size, fill_value=255.).astype('float')
            image1 /= 255
            image2 /= 255
            image3 /= 255
            return [image1, image2, image3]
    elif config['output_masks'] == 3 and config['input_images'] == 2:
        x1_scale = int(x - patch_size // 2)  # 128
        y1_scale = int(y - patch_size // 2)
        x_patch, y_patch = patch_size * 2, patch_size * 2

        x1_scale2 = x1_scale if x1_scale > 0 else 0
        y1_scale2 = y1_scale if y1_scale > 0 else 0

        if y1_scale + patch_size * 2 > size_y:
            y_patch = size_y - y1_scale
        if x1_scale + patch_size * 2 > size_x:
            x_patch = size_x - x1_scale

        reader.rdr.setSeries(0)
        image2 = reader.read(0, rescale=False, XYWH=(
            x1_scale2, y1_scale2, x_patch, y_patch)).astype('float')

        if image1.shape[0] != patch_size or image1.shape[1] != patch_size:
            image1 = get_fragment(
                image1, x, y, patch_size, fill_value=255.).astype('float')
        if image2.shape[0] != patch_size * 2 or image2.shape[1] != patch_size * 2:
            image2 = get_fragment(
                image2, x1_scale, y1_scale, patch_size * 2, fill_value=255.).astype('float')

        if lab:
            image1 = cv.cvtColor(
                image1.astype(np.uint8), cv.COLOR_RGB2LAB)
            image2 = cv.cvtColor(
                image2.astype(np.uint8), cv.COLOR_RGB2LAB)
        else:
            image1 /= 255
            image2 /= 255
        return [image1, image2]
    elif config['output_masks'] == 3 and config['input_images'] == 1:
        pass
    elif config['output_masks'] == 4 and config['input_images'] == 3:
        pass


def handle_empty_images(config, patch_size, channels):
    if config['output_masks'] == 3 and config['input_images'] == 2:
        return [
            np.zeros((patch_size, patch_size, channels)),
            np.zeros((patch_size * 2, patch_size * 2, 3))
        ]

    return [
        np.zeros((patch_size, patch_size, channels)),
        np.zeros((patch_size // 2, patch_size // 2, 3))
    ]


def get_batch(config, reader, coordinates, size_x, size_y, cell_mask, index):
    patch_size = config['image_size']
    channels = config['channels']
    use_cells = config['channels'] == 4
    images = []
    image_coors = []

    for i in range(index, len(coordinates)):
        x, y = coordinates[i]
        x_patch, y_patch = patch_size, patch_size
        if y + y_patch > size_y:
            y_patch = size_y - y
        if x + x_patch > size_x:
            x_patch = size_x - x

        if config['type'] == 'srel':
            reader.rdr.setSeries(1)
            image1 = reader.read(0, rescale=False, XYWH=(
                x, y, x_patch, y_patch)).astype('float')
        else:
            reader.rdr.setSeries(0)
            image1 = reader.read(0, rescale=False, XYWH=(
                x, y, x_patch, y_patch)).astype('float')

        _, std = cv.meanStdDev(image1)
        if std.mean() >= 15:
            images.append(handle_data(reader, image1, config, (x, y,
                          patch_size, patch_size), cell_mask if use_cells else None, (size_x, size_y)))
            image_coors.append([x, y])
            if len(images) == config['batch_size']:
                index = i
                break

    if len(images) < config['batch_size']:
        for i in range(len(images), config['batch_size']):
            images.append(handle_empty_images(config, patch_size, channels))
            image_coors.append([-1, -1])

        return images, -1, image_coors

    return images, index + 1, image_coors


def infere(models):
    if len(models) == 1:
        return infere_nested
    else:
        return infere_deeplab


def infere_deeplab(models, configs, images, canvas, divim, image_coors):
    patch_size = configs[0]['image_size']
    if len(images) == 0:
        return canvas, divim

    dataset = Dataset(
        images=images,
        config=configs[0]
    )

    index = 0
    for patch in dataset:
        img, _ = patch
        pred_mask = models[0].predict(img)

        pred_mask1 = models[1].predict(img)

        pred_mask2 = models[2].predict(img)

        final_mask = np.zeros(
            (pred_mask.shape[1], pred_mask.shape[2], len(configs[0]['classes'])))

        for idx in range(pred_mask.shape[0]):
            mask1 = pred_mask[idx]
            mask2 = pred_mask1[idx]
            mask3 = pred_mask2[idx]

            predicted = np.argmax(mask1, axis=-1)
            predicted1 = np.argmax(mask2, axis=-1)
            predicted2 = np.argmax(mask3, axis=-1)

            mask1 = tf.one_hot(predicted, len(configs[0]['final_classes']))
            mask2 = tf.one_hot(predicted1, len(configs[0]['final_classes']))
            mask3 = tf.one_hot(predicted2, len(configs[0]['final_classes']))

            mask1 = reorder_channels(mask1, configs[0])
            mask2 = reorder_channels(mask2, configs[1])
            mask3 = reorder_channels(mask3, configs[2])

            final_mask[:, :, 0] = cv.bitwise_or(
                mask3[:, :, 1], cv.bitwise_or(mask1[:, :, 1], mask2[:, :, 1]))
            final_mask[:, :, 1] = cv.bitwise_or(
                mask3[:, :, 2], cv.bitwise_or(mask1[:, :, 2], mask2[:, :, 2]))
            final_mask[:, :, 2] = cv.bitwise_or(
                mask3[:, :, 3], cv.bitwise_or(mask1[:, :, 3], mask2[:, :, 3]))

            x, y = image_coors[index]
            if x == -1 and y == -1:
                break

            mask_patch = final_mask[:patch_size + (canvas.shape[0] - y - final_mask.shape[0]),
                                    : patch_size + (canvas.shape[1] - x - final_mask.shape[1]), :]
            canvas[y: y + final_mask.shape[0], x: x +
                   final_mask.shape[1]] += mask_patch
            divim[y: y + final_mask.shape[0], x: x + final_mask.shape[1]] += 1

            index += 1

    return canvas, divim


def infere_nested(models, configs, images, canvas, divim, image_coors):
    model = models[0]
    horizontal_flip = A.HorizontalFlip(p=1)
    vertical_flip = A.VerticalFlip(p=1)

    hor_ver_flip = A.Compose([
        A.HorizontalFlip(p=1),
        A.VerticalFlip(p=1)
    ])

    patch_size = configs[0]['image_size']
    if len(images) == 0:
        return canvas, divim

    dataset = Dataset(
        images=images,
        config=configs[0]
    )

    index = 0
    batch_size = configs[0]['batch_size']
    for patch in dataset:
        img, mask = patch

        pred_mask = model.predict(img)

        img1, _ = applicate_augmentations(
            horizontal_flip, img, mask, batch_size)
        pred_mask1 = model.predict(img1)

        img2, _ = applicate_augmentations(
            vertical_flip, img, mask, batch_size)
        pred_mask2 = model.predict(img2)

        img3, _ = applicate_augmentations(
            hor_ver_flip, img, mask, batch_size)
        pred_mask3 = model.predict(img3)

        pred_mask1, _ = applicate_augmentations(
            horizontal_flip, pred_mask1, mask, batch_size)
        pred_mask2, _ = applicate_augmentations(
            vertical_flip, pred_mask2, mask, batch_size)
        pred_mask3, _ = applicate_augmentations(
            hor_ver_flip, pred_mask3, mask, batch_size)

        if configs[0]['output_masks'] > 1:
            final_mask1 = np.sum(
                [pred_mask[0], pred_mask1[0], pred_mask2[0], pred_mask3[0]], axis=0) / 4
            final_mask2 = np.sum(
                [pred_mask[1], pred_mask1[1], pred_mask2[1], pred_mask3[1]], axis=0) / 4
            batch = pred_mask[0].shape[0]

        else:
            final_mask1 = np.sum(
                [pred_mask, pred_mask1, pred_mask2, pred_mask3], axis=0) / 4
            batch = pred_mask.shape[0]

        for idx in range(batch):
            patch_size = configs[0]['image_size']
            mask = final_mask1[idx]
            x, y = image_coors[index]
            if x == -1 and y == -1:
                break

            mask_patch = mask[:patch_size + (canvas.shape[0] - y - mask.shape[0]),
                              : patch_size + (canvas.shape[1] - x - mask.shape[1]), :]
            canvas[y: y + mask.shape[0], x: x + mask.shape[1]] += mask_patch
            divim[y: y + mask.shape[0], x: x + mask.shape[1]] += 1

            if configs[0]['output_masks'] > 1:
                patch_size = configs[0]['image_size'] * 2
                mask = final_mask2[idx]
                x1_scale = int(y - patch_size // 2)
                y1_scale = int(x - patch_size // 2)
                x1_scale2 = x1_scale if x1_scale > 0 else 0
                y1_scale2 = y1_scale if y1_scale > 0 else 0

                mask_patch = mask[:patch_size + (canvas.shape[0] - y - mask.shape[0]),
                                  : patch_size + (canvas.shape[1] - x - mask.shape[1]), :]
                canvas[y1_scale2: y1_scale2 + mask.shape[0],
                       x1_scale2: x1_scale2 + mask.shape[1]] += mask_patch
                divim[y1_scale2: y1_scale2 + mask.shape[0],
                      x1_scale2: x1_scale2 + mask.shape[1]] += 1

            index += 1

    return canvas, divim


def process_image(model_config_path, image_path, output_path, cell_path, metadata_path, overlap=False):
    file_name = image_path.replace(
        '\\', '/').split('/')[-1].split('.')[0].replace('.vsi', '')

    configs = []
    for config_path in model_config_path:
        configs.append(read_config(config_path))

    models = get_model(configs)
    logger.info(f'Models were loaded')

    image_reader = bf.formatreader.make_image_reader_class()()
    image_reader.allowOpenToCheckType(True)
    image_reader.setId(image_path)
    image_reader.setSeries(0)  # tu sa mení series
    wrapper = bf.formatreader.ImageReader(path=image_path, perform_init=False)
    wrapper.rdr = image_reader

    size_x = wrapper.rdr.getSizeX()
    size_y = wrapper.rdr.getSizeY()
    logger.info(f'Image size: {size_x}x{size_y}')

    patch_size = configs[0]['image_size']
    if configs[0]['type'] == 'srel':
        original_x, original_y = size_x, size_y
        size_x, size_y = size_x // 2, size_y // 2

    x_patches = int(math.ceil(size_x / patch_size))
    y_patches = int(math.ceil(size_y / patch_size))
    y_patch = 0
    x_patch = 0

    coordinates = []
    index = 0
    cell_mask = None
    if exists(f'{metadata_path}/metadata.json'):
        with open(f'{metadata_path}/metadata.json') as f:
            metadata = json.load(f)
            coordinates = metadata['coordinates']
            index = metadata['index']

        divim = np.load(f'{metadata_path}/divim.npy', mmap_mode='r+')
        canvas = np.load(f'{metadata_path}/canvas.npy', mmap_mode='r+')
        if exists(f'{metadata_path}/cell_mask.npy'):
            cell_mask = np.load(
                f'{metadata_path}/cell_mask.npy', mmap_mode='r')
            gdf_tissue = gpd.read_file(f'{metadata_path}/tissue.geojson')
            gdf_immune = gpd.read_file(f'{metadata_path}/immune_cells.geojson')
    else:
        canvas = np.zeros((size_y, size_x, len(
            configs[0]['classes'])), dtype='float32')
        divim = np.zeros((size_y, size_x, len(configs[0]['classes'])), 'uint8')

    if len(coordinates) == 0:
        if overlap:
            step = patch_size // 2

            x_patches += (x_patches - 1) * (patch_size / step - 1)
            y_patches += (y_patches - 1) * (patch_size / step - 1)

            for y_patch in range(int(y_patches)):
                for x_patch in range(int(x_patches)):
                    # get x and y coors of the start of the patch
                    x = x_patch * step
                    y = y_patch * step

                    coordinates.append([x, y])
        else:
            for y_patch in range(int(y_patches)):
                for x_patch in range(int(x_patches)):
                    # get x and y coors of the start of the patch
                    x = x_patch * patch_size
                    y = y_patch * patch_size

                    coordinates.append([x, y])

    logger.info(f'Number of patches: {len(coordinates)}')

    if cell_mask is None:
        if cell_path:
            cell_mask, gdf_tissue, gdf_immune = get_cells(
                cell_path, (size_y, size_x))
            gdf_tissue.to_file(
                f'{metadata_path}/tissue.geojson', driver='GeoJSON')
            gdf_immune.to_file(
                f'{metadata_path}/immune_cells.geojson', driver='GeoJSON')
            np.save(f'{metadata_path}/cell_mask.npy', cell_mask)
        else:
            cell_mask = None
    logger.info(f'Cell mask was created')

    infere_method = infere(models)
    try:
        while index != -1:
            images, index, image_coors = get_batch(
                configs[0], wrapper, coordinates,  size_x, size_y, cell_mask, index
            )

            canvas, divim = infere_method(
                models=models,
                configs=configs,
                images=images,
                canvas=canvas,
                divim=divim,
                image_coors=image_coors,
            )

    except Exception as e:
        print(e)
        np.save(f'{metadata_path}/canvas.npy', canvas)
        np.save(f'{metadata_path}/divim.npy', divim)
        json.dump({
            'coordinates': coordinates,
            'index': index,
        }, open(f'{metadata_path}/metadata.json', 'w'))
        javabridge.kill_vm()
        sys.exit(-1)

    divim[divim == 0] = 1
    canvas /= divim
    if configs[0]['model'] == 'MultiScaleAttUnet':
        canvas[:, :, 1:] = np.array(
            canvas[:, :, 1:] > configs[0]['threshold'], dtype='uint8')
        canvas[:, :, 0] = np.array(
            canvas[:, :, 0] > configs[0]['threshold']*2, dtype='uint8')
    else:
        canvas = np.array(canvas > configs[0]['threshold'], dtype='uint8')

    if configs[0]['type'] == 'srel':
        canvas = cv.resize(canvas, (original_x, original_y))

    canvas = postprocess_mask(canvas.astype('uint8'), dilate=True)

    if configs[0]['type'] == 'srel':
        geojson_file = create_geojson(canvas, [
            "endocariums"
        ])
    else:
        canvas = merge_nearest_endocards(canvas)
        if configs[0]['model'] == 'MultiScaleAttUnet':
            geojson_file = create_geojson(canvas, [
                "blood_vessels",
                "endocariums",
                "inflammations"
            ])
        else:
            geojson_file = create_geojson(canvas, [
                "blood_vessels",
                "inflammations",
                "endocariums"
            ])

    gdf = gpd.GeoDataFrame.from_features(geojson_file)
    gdf = postprocess_endocard(gdf, gdf_tissue)
    gdf = postprocess_inflammation(gdf, gdf_immune)
    gdf = postprocess_vessels(gdf, gdf_tissue)

    gdf.to_file(f'{output_path}/{file_name}.geojson', driver='GeoJSON')
    logger.info('GeoJSON was saved')


if __name__ == '__main__':
    model_config_path = 'nested_final.yaml'
    image_name = '1225_21_HE'
    image_path = f'D:/Master Thesis/Data/EMB-IKEM-2022-03-09/{image_name}.vsi'
    cell_path = f'D:/Master Thesis/Data/Cell Annotations/{image_name}.vsi - 20x.geojson'
    output_path = 'D:/Test'
    javabridge.start_vm(class_path=bf.JARS)
    set_logs()

    parser = argparse.ArgumentParser(
        description='Predict higher morphological structures')
    parser.add_argument('--model_config_path', nargs='+', default=['deeplabv3plus1_final.yaml', 'deeplabv3plus2_final.yaml', 'deeplabv3plus3_final.yaml'],
                        help='Path to the model configs', )
    parser.add_argument('--image_path', type=str,
                        help='Path to the image', default=image_path)
    parser.add_argument('--cell_path', type=str,
                        help='Path to the cell mask', default=cell_path)
    parser.add_argument('--output_path', type=str,
                        help='Path to the output folder', default=output_path)
    parser.add_argument('--metadata_path', type=str,
                        help='Path to the metadata folder', default=f'{output_path}/temp')
    parser.add_argument('--overlap', type=bool, default=True)

    args = parser.parse_args()

    process_image(
        model_config_path=args.model_config_path,
        image_path=args.image_path,
        output_path=args.output_path,
        cell_path=args.cell_path,
        metadata_path=args.metadata_path,
        overlap=args.overlap
    )

    javabridge.kill_vm()
