import cv2 as cv
from geojson import Polygon, FeatureCollection, Feature
import geojson
import geopandas as gpd
import json
import numpy as np
import yaml


def read_config(config_name):
    if '/' not in config_name and '\\' not in config_name:
        config_name = f'configs/{config_name}'

    with open(config_name, 'r') as f:
        if config_name.endswith('.json'):
            config = json.load(f)
        else:
            config = yaml.safe_load(f)

    return config


def normalize_lab(image):
    image[:, :, 0] /= 100
    image[:, :, 1] += 128
    image[:, :, 1] /= 255
    image[:, :, 2] += 128
    image[:, :, 2] /= 255

    return image


def get_gauss(image_size):
    mid = image_size // 2
    if image_size == 256:
        step = 0.006
    else:
        step = 0.003

    mat = np.zeros((image_size, image_size))
    for i in range(image_size):
        for j in range(image_size):
            dist = max(abs(i-mid), abs(j-mid))
            mat[i, j] = max(1 - dist*step, 0)

    return np.expand_dims(mat, -1)


def reorder_channels(mask, config):
    new_mask = np.zeros_like(mask)

    if config['classes'] == ["blood_vessels", "inflammations", "endocariums"]:
        new_mask = mask.numpy()
    elif config['classes'] == ["inflammations", "blood_vessels", "endocariums"]:
        new_mask[:, :, 0] = mask[:, :, 0]
        new_mask[:, :, 1] = mask[:, :, 2]
        new_mask[:, :, 2] = mask[:, :, 1]
        new_mask[:, :, 3] = mask[:, :, 3]

    elif config['classes'] == ["endocariums", "blood_vessels", "inflammations"]:
        new_mask[:, :, 0] = mask[:, :, 0]
        new_mask[:, :, 1] = mask[:, :, 2]
        new_mask[:, :, 2] = mask[:, :, 3]
        new_mask[:, :, 3] = mask[:, :, 1]

    return new_mask


def applicate_augmentations(aug, img, mask, batch_size):
    is_list = True
    if type(img) is not list:
        is_list = False
        img = [img]
        mask = [mask]

    output_img = []
    for image_idx in range(len(img)):
        new_img = np.zeros_like(img[image_idx])
        for idx_batch in range(batch_size):
            augmented = aug(image=img[image_idx][idx_batch],
                            mask=mask[image_idx][idx_batch])
            new_img[idx_batch] = augmented['image']
            mask[image_idx][idx_batch] = augmented['mask']
        output_img.append(new_img)

    if not is_list:
        output_img = output_img[0]
        mask = mask[0]
    return output_img, mask


def get_fragment(image, x, y, patch_size, fill_value=255.):
    if len(image.shape) == 3:
        height, width, channels = image.shape
        canvas = np.full((patch_size, patch_size, channels),
                         fill_value=fill_value)
    else:
        height, width = image.shape
        canvas = np.full((patch_size, patch_size), fill_value=fill_value)

    try:
        if x < 0 and y < 0 and width != patch_size and height != patch_size:  # DONE
            canvas[abs(y):abs(y) + height, abs(x): abs(x) +
                   width] = image[0:height, 0:width]
        elif x < 0 and y < 0 and width != patch_size:  # DONE
            canvas[abs(y):patch_size, abs(x):abs(x) +
                   width] = image[0:patch_size, 0:width]
        elif x < 0 and y < 0 and height != patch_size:  # DONE
            canvas[abs(y):abs(y) + height, abs(x):patch_size] = image[0:height, 0:patch_size]
        elif x < 0 and y < 0:  # DONE
            canvas[abs(y):patch_size, abs(
                x):patch_size] = image[0:patch_size, 0:patch_size]
        elif x < 0 and width != patch_size and height != patch_size:  # DONE
            canvas[0:height, abs(x): abs(x) + width] = image[0:height, 0:width]
        elif x < 0 and width != patch_size:  # DONE
            canvas[:, abs(x):abs(x) + width] = image[0:patch_size, 0:width]
        elif y < 0 and height != patch_size and width != patch_size:  # DONE
            canvas[abs(y):abs(y) + height, 0:width] = image[0:height, 0:width]
        elif y < 0 and width != patch_size:  # DONE
            canvas[abs(y):patch_size, 0:width] = image[0:patch_size, 0:width]
        elif y < 0 and height != patch_size:  # DONE
            canvas[abs(y):abs(y) + height,
                   :] = image[0:height, 0:patch_size]
        elif y < 0:
            canvas[abs(y):patch_size, :] = image[0:patch_size, 0:patch_size]
        elif width != patch_size and height != patch_size:
            canvas[0:height, 0:width] = image[0:height, 0:width]
        elif x < 0 and height != patch_size:
            canvas[0:height, abs(x):patch_size] = image[0:height, 0:patch_size]
        elif height != patch_size:
            canvas[0: height, :] = image[0:height, 0:patch_size]
        elif width != patch_size:
            canvas[:, 0: width] = image[0:patch_size, 0:width]
        elif x < 0:
            canvas[:, abs(x):patch_size] = image[0:patch_size, 0:patch_size]
        else:
            canvas[:, :] = image[0:patch_size, 0:patch_size]
    except Exception as e:
        print(e)
        print(f"X: {x}, Y: {y}, patch_size: {patch_size}")
        raise Exception(e)

    return canvas


def get_cells(cell_path, shape):
    gj = geojson.load(open(cell_path))

    x, y = int(shape[0]), int(shape[1])

    mask = np.zeros((x, y), dtype='uint8')
    immune_cells = []

    for feat in gj['features']:
        if feat['properties'].get('classification', None) and feat['properties']['classification']['name'] == 'Region*':
            tissues = feat['geometry']['coordinates']
            len_fragments = len(tissues)
            polygons = []
            for i in range(len_fragments):
                coors = tissues[i][0]
                polygons.append(Feature(
                    geometry=Polygon([coors]),
                ))

            gdf_tissue = gpd.GeoDataFrame.from_features(
                FeatureCollection(polygons))
        elif feat['properties'].get('classification', None) is None or feat['properties']['classification']['name'] != 'Region*':
            geometry_name = 'nucleusGeometry' if feat.get(
                'nucleusGeometry') else 'geometry'
            coors = feat[geometry_name]['coordinates'][0]
            pts = [[round(c[0]), round(c[1])] for c in coors]
            cv.fillPoly(
                mask,
                [np.array(pts)],
                1
            )

            if feat['properties']['classification']['name'] == 'Immune cells':
                immune_cells.append(Feature(
                    geometry=Polygon([coors]),
                ))

    gdf_imunne = gpd.GeoDataFrame.from_features(
        FeatureCollection(immune_cells))
    return mask, gdf_tissue, gdf_imunne
